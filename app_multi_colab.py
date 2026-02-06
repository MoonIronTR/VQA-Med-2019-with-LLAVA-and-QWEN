import os
import torch
import streamlit as st
import pandas as pd
from PIL import Image
import glob
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import gc
import json
import time
from huggingface_hub import login

from peft import PeftModel
from transformers import (
    LlavaForConditionalGeneration, LlavaProcessor,
    InstructBlipForConditionalGeneration, InstructBlipProcessor,
    PaliGemmaForConditionalGeneration, AutoProcessor,
    BitsAndBytesConfig
)

# --- NLTK Setup (app.py aynısı) ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Page Config (app.py aynısı)
st.set_page_config(
    page_title="Medical VQA Assistant",
    page_icon="🏥",
    layout="wide"
)

# ================= DRIVE AYARLARI =================
BASE_PATH = "/content/drive/MyDrive/VQA_Multi"

# Modellerin Klasörleri (Senin yapına göre)
MODEL_CONFIGS = {
    "LLaVA-v1.5-7b": {
        "base_id": "llava-hf/llava-1.5-7b-hf",
        "adapter_path": f"{BASE_PATH}/models/llava",
        "type": "llava"
    },
    "InstructBLIP-Flan-T5-XL": {
        "base_id": "Salesforce/instructblip-flan-t5-xl",
        "adapter_path": f"{BASE_PATH}/models/instructblip",
        "type": "instructblip"
    },
    "PaliGemma-3b-Mix": {
        "base_id": "google/paligemma-3b-mix-224",
        "adapter_path": f"{BASE_PATH}/models/paligemma",
        "type": "paligemma"
    }
}

TEST_TXT_PATH = f"{BASE_PATH}/dataset/Test_Questions_w_Ref_Answers.txt"
TEST_IMG_DIR = f"{BASE_PATH}/dataset/test_images"
SYSTEM_PROMPT = "You are a medical VQA assistant." # app.py'daki Prompt

# ================= METRICS LOGIC (app.py'dan BİREBİR KOPYA) =================
STOPWORDS = set(stopwords.words("english")) - {"yes", "no"}
STEMMER = SnowballStemmer("english")
SMOOTH = SmoothingFunction().method1

def sanitize_text(s):
    if not s: return ""
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_text_basic(text):
    text = sanitize_text(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_and_tokenize(text):
    text = clean_text_basic(text)
    return [STEMMER.stem(t) for t in text.split() if t not in STOPWORDS]

def is_yes_no_strict(s):
    s = sanitize_text(s).lower().strip()
    if s == "yes": return "yes"
    if s == "no": return "no"
    return None

def expand_reference_answers(gt):
    gt = sanitize_text(gt)
    if not gt: return [""]
    parts = [p.strip() for p in gt.split("#") if p.strip()]
    if not parts: parts = [gt.strip()]
    refs_set = set()
    for p in parts:
        refs_set.add(p)
        dropped = re.sub(r"\([^)]*\)", "", p)
        dropped = re.sub(r"\s+", " ", dropped).strip()
        if dropped and dropped != p: refs_set.add(dropped)
    return list(refs_set)

def calculate_single_score(pred, ground_truth):
    refs_text = expand_reference_answers(ground_truth)
    gt_yesno_vals = {is_yes_no_strict(r) for r in refs_text if is_yes_no_strict(r) is not None}
    is_yesno_question = len(gt_yesno_vals) > 0
    
    pred_norm = clean_text_basic(pred)
    refs_norm = [clean_text_basic(r) for r in refs_text]
    
    exact_match = False
    if is_yesno_question:
        pred_yn = is_yes_no_strict(pred)
        if pred_yn is not None and pred_yn in gt_yesno_vals: exact_match = True
    else:
        exact_match = any(pred_norm == r for r in refs_norm if r)
        
    pred_tok = normalize_and_tokenize(pred)
    refs_tok_list = [normalize_and_tokenize(r) for r in refs_text if normalize_and_tokenize(r)]
    
    token_match = False
    if is_yesno_question: token_match = exact_match
    else: token_match = any(pred_tok == rt for rt in refs_tok_list)
        
    bleu = 0.0
    if pred_tok and refs_tok_list:
        bleu = sentence_bleu(refs_tok_list, pred_tok, (1, 0, 0, 0), smoothing_function=SMOOTH)
        
    return {
        "exact_match": exact_match,
        "token_match": token_match,
        "is_yesno": is_yesno_question,
        "bleu": bleu,
        "normalized_pred": " ".join(pred_tok),
        "normalized_refs": [" ".join(r) for r in refs_tok_list]
    }

# ================= YARDIMCI FONKSİYONLAR =================
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def find_adapter_dir(search_path: str) -> str:
    if os.path.exists(os.path.join(search_path, "adapter_config.json")): return search_path
    final = os.path.join(search_path, "final_adapter")
    if os.path.isdir(final): return final
    ckpts = sorted(glob.glob(os.path.join(search_path, "checkpoint-*")))
    if ckpts:
        ckpts.sort(key=lambda p: int(p.split("-")[-1]))
        return ckpts[-1]
    return search_path if os.path.isdir(search_path) else ""

def resize_image_for_inference(image):
    max_size = 1024 
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    return image

@st.cache_resource(show_spinner=False)
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    )

def load_model_pipeline(model_key):
    clear_gpu_memory() 
    
    config = MODEL_CONFIGS[model_key]
    base_id = config["base_id"]
    adapter_path_raw = config["adapter_path"]
    m_type = config["type"]
    
    status = st.empty()
    status.info(f"Loading {model_key} base model...")
    
    try:
        load_kwargs = {
            "quantization_config": get_bnb_config(),
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True
        }
        
        # Model Yükleme
        if m_type == "llava":
            model = LlavaForConditionalGeneration.from_pretrained(base_id, **load_kwargs)
            processor = LlavaProcessor.from_pretrained(base_id)
        elif m_type == "instructblip":
            model = InstructBlipForConditionalGeneration.from_pretrained(base_id, **load_kwargs)
            processor = InstructBlipProcessor.from_pretrained(base_id)
        elif m_type == "paligemma":
            model = PaliGemmaForConditionalGeneration.from_pretrained(base_id, **load_kwargs)
            processor = AutoProcessor.from_pretrained(base_id)
        
        real_adapter = find_adapter_dir(adapter_path_raw)
        if real_adapter:
            status.info(f"Loading Adapter: {real_adapter}...")
            model = PeftModel.from_pretrained(model, real_adapter)
        else:
            st.warning(f"Adapter not found in {adapter_path_raw}. Using Base Model.")
            
        model.eval()
        status.empty()
        return model, processor, m_type
        
    except Exception as e:
        status.error(f"Error loading model: {e}")
        clear_gpu_memory()
        return None, None, None

def load_test_data(txt_path):
    rows = []
    if not os.path.exists(txt_path): return None
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 4:
                rows.append({"image_id": parts[0].strip(), "category": parts[1].strip(), "question": parts[2].strip(), "answer": parts[3].strip()})
    return pd.DataFrame(rows)

def clean_pred(text: str) -> str:
    if not text: return ""
    t = text.strip().replace("\u200b", "")
    if "\n" in t: t = t.splitlines()[0]
    for bad in ["ASSISTANT:", "Assistant:", "assistant:"]:
        if t.startswith(bad): t = t[len(bad):].strip()
    return t

# ================= INFERENCE LOGIC (Burada Ayrım Yapıyoruz) =================
def run_inference(model, processor, m_type, image, question):
    try:
        device = model.device
        image = resize_image_for_inference(image)
        
        inputs = None
        
        # 1. LLaVA (app.py mantığı korunuyor)
        if m_type == "llava":
            prompt = f"SYSTEM: {SYSTEM_PROMPT}\nUSER: <image>\nQuestion: {question}\nAnswer briefly with a short phrase.\nASSISTANT:"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            gen_kwargs = {"max_new_tokens": 40, "do_sample": False}

        # 2. InstructBLIP (Notebook mantığına geçiyoruz - Düzgün çalışması için)
        elif m_type == "instructblip":
            prompt = f"You are a medical assistant. Analyze this radiology image. Question: {question} Answer:"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            gen_kwargs = {"max_new_tokens": 30, "min_length": 1, "do_sample": False, "num_beams": 3}

        # 3. PaliGemma (Standart mantık)
        elif m_type == "paligemma":
            prompt = f"answer {question}"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            gen_kwargs = {"max_new_tokens": 40, "do_sample": False}

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            
        # Decode
        if m_type == "llava":
            pred = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        elif m_type == "paligemma":
            pred = processor.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        else:
            pred = processor.decode(outputs[0], skip_special_tokens=True)
            
        return clean_pred(pred)
        
    except torch.cuda.OutOfMemoryError:
        clear_gpu_memory()
        return "ERROR_OOM"
    except Exception as e:
        return f"ERROR: {str(e)}"

# ================= BATCH HELPERS =================
def extract_json_pairs(obj):
    pairs = []
    def _get_cat(item): return item.get("category", "unknown")
    
    if isinstance(obj, list):
        for s in obj:
            if isinstance(s, dict): pairs.append((s.get("ground_truth", ""), s.get("predicted", ""), _get_cat(s)))
            
    elif isinstance(obj, dict):
        if "samples" not in obj:
            for img_id, qa_list in obj.items():
                if isinstance(qa_list, list):
                    for qa in qa_list:
                        pairs.append((qa.get("ground_truth", ""), qa.get("predicted", ""), _get_cat(qa)))
        else:
             if isinstance(obj["samples"], list):
                for s in obj["samples"]:
                    if "qas" in s:
                         for qa in s["qas"]: pairs.append((qa.get("ground_truth", ""), qa.get("predicted", ""), _get_cat(qa)))
                    else: pairs.append((s.get("ground_truth", ""), s.get("predicted", ""), _get_cat(s)))
    return pairs

def compute_batch_stats(pairs):
    stats = []
    for gt, pred, cat in pairs:
        s = calculate_single_score(pred, gt)
        stats.append({
            "category": cat,
            "acc_strict": 1 if s["exact_match"] else 0,
            "acc_norm": 1 if (s["exact_match"] if s["is_yesno"] else s["token_match"]) else 0,
            "bleu": s["bleu"]
        })
    return pd.DataFrame(stats)

# ================= BATCH GENERATION (JSON Üretici) =================
def run_batch_generation(df, model, processor, m_type, img_dir):
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    unique_ids = df['image_id'].unique()
    total_imgs = len(unique_ids)
    
    for idx, img_id in enumerate(unique_ids):
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path): img_path = os.path.join(img_dir, f"{img_id}.png")
        if not os.path.exists(img_path): continue
            
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception: continue

        subset = df[df['image_id'] == img_id]
        img_results = []
        
        for _, row in subset.iterrows():
            question = row['question']
            gt = row['answer']
            cat = row['category']
            
            pred = run_inference(model, processor, m_type, image, question)
            
            img_results.append({
                "category": cat,
                "question": question,
                "ground_truth": gt,
                "predicted": pred
            })
            
        results[img_id] = img_results
        
        if idx % 10 == 0:
            status_text.text(f"Processing: {idx}/{total_imgs} - ID: {img_id}")
            progress_bar.progress((idx + 1) / total_imgs)
            
    progress_bar.progress(1.0)
    status_text.text("Completed!")
    return results

# ================= UI (app.py Tasarımı Aynen Korundu) =================
st.title("🏥 Medical VQA Assistant")

with st.sidebar:
    st.header("⚙️ Configuration")
    app_mode = st.radio("Mode", ["Single Image Inference", "Batch Metrics Dashboard"], index=0)
    st.divider()
    
    # Model Seçimi (app.py'dan tek farkı bu kutu, Colab için gerekli)
    model_key = st.selectbox("Select Model", list(MODEL_CONFIGS.keys()))
    
    if st.button("Load/Reload Model"):
        if "pipeline" in st.session_state and st.session_state.pipeline:
            del st.session_state.pipeline
            st.session_state.pipeline = None
            clear_gpu_memory()
        
        st.session_state.pipeline = load_model_pipeline(model_key)
        st.session_state.current_model = model_key
    
    if "pipeline" in st.session_state and st.session_state.pipeline:
        st.success(f"Loaded: {st.session_state.current_model}")
        if torch.cuda.is_available(): st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("No model loaded.")

# --- MODE 1: SINGLE IMAGE ---
if app_mode == "Single Image Inference":
    if "pipeline" not in st.session_state or not st.session_state.pipeline:
        st.info("👈 Please load a model from the sidebar.")
        st.stop()
        
    model, processor, m_type = st.session_state.pipeline
    
    try:
        df_test = load_test_data(TEST_TXT_PATH)
        valid_ids = set(df_test["image_id"].values)
    except:
        df_test = None
        valid_ids = set()

    col_img, col_qa = st.columns([1, 1.2])

    with col_img:
        st.subheader("1. Upload Image")
        uploaded_file = st.file_uploader("Drop an image here...", type=["jpg", "png", "jpeg"])
        image = None
        file_id = None
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_column_width=True)
            file_id = os.path.splitext(uploaded_file.name)[0]
            if file_id in valid_ids: st.success(f"✅ Recognized ID: `{file_id}`")
            else: st.info("Custom Image")

    with col_qa:
        st.subheader("2. Select Question")
        question = ""
        ground_truth = None
        
        if file_id and file_id in valid_ids and df_test is not None:
            subset = df_test[df_test["image_id"] == file_id]
            q_mode = st.radio("Question Mode", ["Select Valid Question", "Custom Question"], horizontal=True)
            if q_mode == "Select Valid Question":
                q_map = {row["question"]: row["answer"] for _, row in subset.iterrows()}
                question = st.selectbox("Available Questions:", list(q_map.keys()))
                ground_truth = q_map.get(question)
                if question:
                    cat = subset[subset['question'] == question]['category'].values[0]
                    st.caption(f"Category: {cat}")
            else:
                question = st.text_input("Type your question:")
        else:
            st.warning("Model trained on: Organ, Modality, Plane, Abnormality.")
            question = st.text_input("Type your question:", placeholder="e.g. What organ is this?")

        st.divider()
        if st.button("🔍 Predict Answer", disabled=(not image or not question)):
            with st.spinner("Analyzing..."):
                pred_clean = run_inference(model, processor, m_type, image, question)
                
                if "ERROR" in pred_clean:
                    st.error(pred_clean)
                else:
                    st.markdown("### 🤖 Prediction")
                    st.success(f"**{pred_clean}**")
                    
                    if ground_truth:
                        st.markdown("### ✅ Ground Truth")
                        st.info(f"**{ground_truth}**")
                        
                        # Skor Hesaplama (app.py ile aynı)
                        scores = calculate_single_score(pred_clean, ground_truth)
                        m1, m2 = st.columns(2)
                        
                        lbl = "Accuracy (Strict)" if scores["is_yesno"] else "Accuracy (Normalized)"
                        val = scores["exact_match"] if scores["is_yesno"] else scores["token_match"]
                        
                        if val:
                            m1.metric(lbl, "Correct", delta="Match", delta_color="normal")
                            st.balloons()
                        else:
                            m1.metric(lbl, "Incorrect", delta="Mismatch", delta_color="inverse")
                        m2.metric("BLEU-1 Score", f"{scores['bleu']:.4f}")
                        
                        with st.expander("Details"):
                            st.write(f"Norm Pred: `{scores['normalized_pred']}`")
                            st.write(f"Norm Ref: `{scores['normalized_refs']}`")

# --- MODE 2: BATCH METRICS ---
elif app_mode == "Batch Metrics Dashboard":
    st.header("📊 Batch Evaluation")
    
    tab1, tab2 = st.tabs(["Run Test Set (Generate JSON)", "Analyze Existing JSON"])
    
    # TAB 1: GENERATE
    with tab1:
        st.write("Runs the currently loaded model on the full test set.")
        if "pipeline" not in st.session_state or not st.session_state.pipeline:
            st.warning("Please load a model first.")
        else:
            if not os.path.exists(TEST_IMG_DIR):
                st.error(f"Image directory not found: {TEST_IMG_DIR}")
            else:
                df_test = load_test_data(TEST_TXT_PATH)
                if df_test is not None:
                    st.write(f"Total Questions: {len(df_test)}")
                    if st.button("🚀 Start Batch Inference"):
                        model, processor, m_type = st.session_state.pipeline
                        with st.spinner("Running batch inference..."):
                            res = run_batch_generation(df_test, model, processor, m_type, TEST_IMG_DIR)
                        
                        json_str = json.dumps(res, indent=2)
                        st.success("Done!")
                        st.download_button("📥 Download JSON", json_str, file_name=f"preds_{st.session_state.current_model}.json", mime="application/json")

    # TAB 2: ANALYZE
    with tab2:
        u_json = st.file_uploader("Upload Predictions JSON", type=["json"])
        if u_json:
            try:
                pairs = extract_json_pairs(json.load(u_json))
                if pairs:
                    df = compute_batch_stats(pairs)
                    acc_strict = df["acc_strict"].mean() * 100
                    acc_norm = df["acc_norm"].mean() * 100
                    bleu_avg = df["bleu"].mean()
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Accuracy (Strict)", f"{acc_strict:.2f}%")
                    m2.metric("Accuracy (Norm)", f"{acc_norm:.2f}%")
                    m3.metric("BLEU-1", f"{bleu_avg:.4f}")
                    
                    st.subheader("Category Breakdown")
                    cat_grp = df.groupby("category")[["acc_norm", "bleu"]].mean().reset_index()
                    cat_grp["acc_norm"] = cat_grp["acc_norm"] * 100
                    st.dataframe(cat_grp.style.format({"acc_norm": "{:.2f}%", "bleu": "{:.4f}"}))
            except Exception as e:
                st.error(f"Error: {e}")