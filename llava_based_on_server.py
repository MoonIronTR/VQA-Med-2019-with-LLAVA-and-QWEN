import os
import time
import json
import glob
from datetime import timedelta
import sys
from functools import lru_cache

import pandas as pd
import torch
from PIL import Image as PImage

from datasets import Dataset
import transformers
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

import peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ================== UTIL ==================

def safe_version(name, obj):
    ver = getattr(obj, "__version__", None)
    print(f"{name} version:", ver if ver is not None else "bulunamadı")

def get_env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "")
    if v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def read_pipe_qa(path_txt: str) -> pd.DataFrame:
    """
    TXT'ye dokunmadan, satır satır okuyup `split("|", 2)` ile ayırıyoruz.
    (image_id | question | answer) -> sadece ilk 2 pipe ayrılır.
    Böylece içerdeki " karakterleri parsing'i bozmaz.
    """
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            rows = []
            with open(path_txt, "r", encoding=enc, errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.rstrip("\n\r")
                    if not line.strip():
                        continue
                    parts = line.split("|", 2)
                    if len(parts) < 3:
                        continue
                    image_id = parts[0].strip()
                    question = parts[1].strip()
                    answer   = parts[2].strip()
                    rows.append((image_id, question, answer))
            return pd.DataFrame(rows, columns=["image_id", "question", "answer"])
        except Exception as e:
            last_err = e
    raise RuntimeError(f"QA dosyası okunamadı: {path_txt} (son hata: {last_err})")


# ================== SPEED KNOBS (SAFE) ==================

# Ampere+ ise genelde iyi hız kazancı:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ================== VERSIONS + GPU ==================

print("==== Python ve Temel Bilgiler ====")
print("Python:", sys.version)
print()

print("==== Ana Kütüphaneler ====")
safe_version("pandas", pd)
safe_version("torch", torch)
safe_version("transformers", transformers)
safe_version("peft", peft)

try:
    import bitsandbytes as bnb
    safe_version("bitsandbytes", bnb)
except ImportError:
    bnb = None
    print("bitsandbytes: yüklü değil (4-bit kapalı çalışacak)")

print("\n==== GPU Durumu ====")
print("torch.cuda.is_available():", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("⚠ GPU bulunamadı. LLaVA 7B GPU olmadan pratik değil.")
    sys.exit(0)
print("GPU:", torch.cuda.get_device_name(0))


# ===================== 1) PATH + PARAMETRELER =====================

BASE_ROOT = os.environ.get("BASE_ROOT", "/home/aaydemir/vqa2019")
LOCAL_DATA_ROOT = os.environ.get(
    "LOCAL_DATA_ROOT",
    os.path.join(BASE_ROOT, "ImageClef-2019-VQA-Med-Training")
)

TRAIN_QA_FILE = os.path.join(LOCAL_DATA_ROOT, "All_QA_Pairs_train.txt")
VAL_QA_FILE   = os.path.join(LOCAL_DATA_ROOT, "All_QA_Pairs_val.txt")

TRAIN_IMAGE_DIR = os.path.join(LOCAL_DATA_ROOT, "Train_images")
VAL_IMAGE_DIR   = os.path.join(LOCAL_DATA_ROOT, "Val_images")
IMAGE_EXTENSION = os.environ.get("IMAGE_EXTENSION", ".jpg")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_ROOT, "llava_ft_out_v2"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ID = os.environ.get("MODEL_ID", "llava-hf/llava-1.5-7b-hf")
SYSTEM_PROMPT = "You are a medical VQA assistant."

# max epoch yüksek kalabilir; early stopping durdurur
NUM_TRAIN_EPOCHS = int(os.environ.get("NUM_TRAIN_EPOCHS", "8"))

USE_4BIT = get_env_bool("USE_4BIT", True) and (bnb is not None)
UNFREEZE_VISION = get_env_bool("UNFREEZE_VISION", False)

print("\n==== PATH KONTROL ====")
print("BASE_ROOT       :", BASE_ROOT)
print("LOCAL_DATA_ROOT :", LOCAL_DATA_ROOT)
print("TRAIN_QA_FILE   :", TRAIN_QA_FILE)
print("VAL_QA_FILE     :", VAL_QA_FILE)
print("TRAIN_IMAGE_DIR :", TRAIN_IMAGE_DIR)
print("VAL_IMAGE_DIR   :", VAL_IMAGE_DIR)
print("OUTPUT_DIR      :", OUTPUT_DIR)
print("MODEL_ID        :", MODEL_ID)
print("USE_4BIT        :", USE_4BIT)
print("EPOCHS(max)     :", NUM_TRAIN_EPOCHS)
print("UNFREEZE_VISION :", UNFREEZE_VISION)

for p in [LOCAL_DATA_ROOT, TRAIN_QA_FILE, VAL_QA_FILE, TRAIN_IMAGE_DIR, VAL_IMAGE_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Bulunamadı: {p}")


# ===================== 2) DATAFRAME -> HF DATASET =====================

def load_qa_file(path_txt: str, image_dir: str, name: str) -> pd.DataFrame:
    print(f"\n🔹 {name} TXT okunuyor:", path_txt)

    df = read_pipe_qa(path_txt)
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"]   = df["answer"].astype(str).str.strip()

    df["image_path"] = df["image_id"].apply(lambda x: os.path.join(image_dir, f"{x}{IMAGE_EXTENSION}"))
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(f"{name} için hiç geçerli örnek yok. Klasör/uzantı yanlış olabilir.")

    print(f"{name} QA sayısı     : {len(df)}")
    print(f"{name} unique images : {df['image_id'].nunique()}")
    return df

df_train = load_qa_file(TRAIN_QA_FILE, TRAIN_IMAGE_DIR, "TRAIN")
df_val   = load_qa_file(VAL_QA_FILE,   VAL_IMAGE_DIR,   "VAL")


# ===================== 3) MODEL + PROCESSOR =====================

bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

processor = LlavaProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer

# Vision tower: default freeze
for p in model.vision_tower.parameters():
    p.requires_grad = False

if UNFREEZE_VISION:
    n_unfreeze = int(os.environ.get("UNFREEZE_LAST_N_PARAMS", "200"))
    vt_params = list(model.vision_tower.parameters())
    for p in vt_params[-n_unfreeze:]:
        p.requires_grad = True
    print(f"✅ Vision tower kısmen açıldı: son {n_unfreeze} parametre grubu")

if USE_4BIT:
    model = prepare_model_for_kbit_training(model)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# LoRA
lora_config = LoraConfig(
    r=int(os.environ.get("LORA_R", "64")),
    lora_alpha=int(os.environ.get("LORA_ALPHA", "128")),
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.05")),
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

print("\nEğitilebilir Parametre Sayısı:")
model.print_trainable_parameters()


# ===================== 4) DATASET (prompt_len PRECOMPUTE) =====================

def build_prompt(q: str) -> str:
    return (
        f"SYSTEM: {SYSTEM_PROMPT}\n"
        f"USER: <image>\n"
        f"Question: {q}\n"
        f"Answer briefly with a short phrase.\n"
        f"ASSISTANT:"
    )

def format_and_load_dataset(df: pd.DataFrame) -> Dataset:
    prompts, prompt_lens, answers = [], [], []
    for _, row in df.iterrows():
        p = build_prompt(row["question"])
        # ✅ prompt_len'i 1 kere hesapla (her batch'te tokenizer çağrısı kalkar)
        pl = len(tokenizer(p, add_special_tokens=False)["input_ids"])
        prompts.append(p)
        prompt_lens.append(pl)
        answers.append(row["answer"])

    return Dataset.from_dict(
        {
            "image_path": df["image_path"].tolist(),
            "image_id": df["image_id"].tolist(),
            "prompt": prompts,
            "prompt_len": prompt_lens,
            "answer": answers,
            "question": df["question"].tolist(),
        }
    )

train_dataset = format_and_load_dataset(df_train)
val_dataset   = format_and_load_dataset(df_val)

print(f"\nEğitim Seti Boyutu : {len(train_dataset)}")
print(f"Validation Seti    : {len(val_dataset)}")


# ===================== 5) FAST IMAGE LOADER (LRU CACHE) =====================

# 3200 train + 500 val -> 4096 cache rahat
IMG_CACHE_SIZE = int(os.environ.get("IMG_CACHE_SIZE", "4096"))

@lru_cache(maxsize=IMG_CACHE_SIZE)
def _load_rgb_cached(path: str):
    # File handle sızıntısı olmaması için copy()
    img = PImage.open(path).convert("RGB")
    return img.copy()


# ===================== 6) DATA COLLATOR (answer-only loss + EOS) =====================

def vqa_data_collator(examples):
    # ✅ cache'li image load
    images  = [_load_rgb_cached(e["image_path"]) for e in examples]
    prompts = [e["prompt"] for e in examples]
    prompt_lens = [int(e["prompt_len"]) for e in examples]

    eos = tokenizer.eos_token or ""
    answers = [(e["answer"].strip() + (eos if eos else "")).strip() for e in examples]

    full_texts = [f"{p} {a}" for p, a in zip(prompts, answers)]

    proc_out = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding="longest",
    )

    input_ids      = proc_out["input_ids"]
    attention_mask = proc_out["attention_mask"]
    pixel_values   = proc_out["pixel_values"]

    labels = input_ids.clone()

    # ✅ prompt_len hazır (tokenizer çağrısı yok)
    for i, pl in enumerate(prompt_lens):
        labels[i, :pl] = -100

    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }


# ===================== 7) TRAINING ARGS (LESS OVERHEAD + FASTER DATALOADER) =====================

# ✅ en büyük hız kazancı: eval/save seyrek
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "1000"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "1000"))

# ✅ dataloader hız ayarları
DL_WORKERS = int(os.environ.get("DL_WORKERS", "4"))  # GPU boşta kalmasın
PREFETCH = int(os.environ.get("PREFETCH_FACTOR", "2"))

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,

    per_device_train_batch_size=int(os.environ.get("BATCH_SIZE", "1")),
    gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "16")),

    per_device_eval_batch_size=1,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=int(os.environ.get("SAVE_TOTAL_LIMIT", "2")),
    save_safetensors=True,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    learning_rate=float(os.environ.get("LR", "2e-5")),
    warmup_ratio=float(os.environ.get("WARMUP_RATIO", "0.05")),
    lr_scheduler_type=os.environ.get("SCHED", "cosine"),

    weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.01")),
    max_grad_norm=float(os.environ.get("MAX_GRAD_NORM", "1.0")),

    optim="adamw_torch",
    fp16=True,

    logging_steps=int(os.environ.get("LOGGING_STEPS", "50")),
    report_to="none",
    remove_unused_columns=False,

    dataloader_num_workers=DL_WORKERS,
    dataloader_pin_memory=True,
)

# persistent_workers/prefetch_factor bazı sürümlerde argüman olarak yok;
# varsa dataloader'a otomatik geçer, yoksa sorun çıkarmaz.
try:
    training_args.dataloader_persistent_workers = True
    training_args.dataloader_prefetch_factor = PREFETCH
except Exception:
    pass

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=vqa_data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=int(os.environ.get("PATIENCE", "3")))],
)

print(f"\n✨ Fine-tuning başlıyor (FAST, epochs(max)={NUM_TRAIN_EPOCHS}, USE_4BIT={USE_4BIT})")
print(f"eval_steps={EVAL_STEPS}, save_steps={SAVE_STEPS}, patience={os.environ.get('PATIENCE','3')}")
print(f"DL_WORKERS={DL_WORKERS}, IMG_CACHE_SIZE={IMG_CACHE_SIZE}")


# ===================== 8) TRAIN + AUTO RESUME =====================

ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
resume_path = ckpts[-1] if ckpts else None

start_time = time.time()

if resume_path is None:
    print("ℹ️ Checkpoint yok -> sıfırdan başlıyor.")
    train_result = trainer.train()
else:
    print(f"ℹ️ Checkpoint bulundu -> devam: {resume_path}")
    train_result = trainer.train(resume_from_checkpoint=resume_path)

end_time = time.time()
total_time_sec = end_time - start_time

metrics = dict(train_result.metrics)
metrics["total_training_time_sec"] = float(total_time_sec)
metrics["total_training_time_str"] = str(timedelta(seconds=int(total_time_sec)))
metrics["num_train_samples"] = len(train_dataset)
metrics["num_eval_samples"] = len(val_dataset)
metrics["num_train_steps"] = train_result.global_step

eval_metrics = trainer.evaluate(val_dataset)
for k, v in eval_metrics.items():
    metrics[f"eval_{k}"] = float(v) if isinstance(v, (int, float)) else v

log_obj = {
    "training_args": training_args.to_dict(),
    "final_metrics": metrics,
    "log_history": trainer.state.log_history,
}

log_path = os.path.join(OUTPUT_DIR, "training_log.json")
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(log_obj, f, indent=2)

print(f"\n📄 Log kaydedildi: {log_path}")
print(f"⏱ Toplam eğitim süresi: {metrics['total_training_time_str']}")

ADAPTER_PATH = os.path.join(OUTPUT_DIR, "final_adapter")
trainer.model.save_pretrained(ADAPTER_PATH)
processor.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Eğitim bitti.")
print(f"✅ Adaptör: {ADAPTER_PATH}")
print(f"✅ En iyi model: load_best_model_at_end=True (eval_loss)")
print(f"✅ Checkpointler: {OUTPUT_DIR}/checkpoint-*")
