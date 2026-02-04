import os
import time
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

from peft import PeftModel
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    BitsAndBytesConfig,
)

# --- NLTK Setup ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Page Config
st.set_page_config(
    page_title="Medical VQA Assistant",
    page_icon="🏥",
    layout="wide"
)

# Constants & Defaults
DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEFAULT_ADAPTER_DIR = r"C:\Users\Arınç\Desktop\bitirme\vqa_downloads\llava_ft_out_v2"
TEST_TXT_PATH = r"C:\Users\Arınç\Desktop\bitirme\vqa2019\VQAMed2019Test\VQAMed2019_Test_Questions_w_Ref_Answers.txt"
TEST_IMG_DIR = r"C:\Users\Arınç\Desktop\bitirme\vqa2019\VQAMed2019Test\VQAMed2019_Test_Images"

SYSTEM_PROMPT = "You are a medical VQA assistant."

# ================= METRICS LOGIC (Portered from compute_metrics.py) =================

STOPWORDS = set(stopwords.words("english")) - {"yes", "no"}
STEMMER = SnowballStemmer("english")
STOPWORDS = set(stopwords.words("english")) - {"yes", "no"}
STEMMER = SnowballStemmer("english")
SMOOTH = SmoothingFunction().method1

def is_yes_no_strict(s):
    """
    Strict yes/no check:
      - Only accepts 'yes' or 'no' (exact match after basic sanitization).
      - Rejects 'yes it is', etc.
    """
    s = sanitize_text(s).lower().strip()
    if s == "yes": return "yes"
    if s == "no": return "no"
    return None

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

def expand_reference_answers(gt):
    """Handles # (alternatives) and () (optional content)"""
    gt = sanitize_text(gt)
    if not gt: return [""]
    
    parts = [p.strip() for p in gt.split("#") if p.strip()]
    if not parts: parts = [gt.strip()]
    
    refs_set = set()
    for p in parts:
        refs_set.add(p)
        # Handle optional parentheses: "(foo) bar" -> "bar"
        dropped = re.sub(r"\([^)]*\)", "", p)
        dropped = re.sub(r"\s+", " ", dropped).strip()
        if dropped and dropped != p:
            refs_set.add(dropped)
            
    return list(refs_set)

def calculate_single_score(pred, ground_truth):
    """Calculates Exact Match (Normalized) and BLEU-1 for a single pair."""
    refs_text = expand_reference_answers(ground_truth)
    
    # --- 1. Accuracy Logic ---
    # Check if this is a Yes/No question based on Ground Truth
    gt_yesno_vals = set()
    for r in refs_text:
        y = is_yes_no_strict(r)
        if y is not None:
            gt_yesno_vals.add(y)
            
    is_yesno_question = len(gt_yesno_vals) > 0
    
    # Normalize for comparison
    pred_norm = clean_text_basic(pred)
    refs_norm = [clean_text_basic(r) for r in refs_text]
    
    exact_match = False
    
    if is_yesno_question:
        # Strict Yes/No Accuracy
        pred_yn = is_yes_no_strict(pred)
        if pred_yn is not None and pred_yn in gt_yesno_vals:
            exact_match = True
    else:
        # Standard Normalized Exact Match for non-Yes/No
        exact_match = any(pred_norm == r for r in refs_norm if r)
    
    # --- 2. Normalized Accuracy (Token Match) ---
    # User requested to use BLEU-style normalization for better accuracy (e.g. "crohn's" == "crohn")
    pred_tok = normalize_and_tokenize(pred)
    refs_tok_list = [normalize_and_tokenize(r) for r in refs_text if normalize_and_tokenize(r)]
    
    # Check if prediction tokens match any of the reference token lists exactly
    # This handles "crohn's disease" -> ['crohn', 'diseas'] vs "crohn disease" -> ['crohn', 'diseas']
    token_match = False
    if is_yesno_question:
         # For Yes/No, stick to strict because stemming "yes" -> "yes" is same, but usually we want strictness there.
         # But technically, if we want "normalized", stemming is consistent. 
         # VQA-Med usually strict for Yes/No. Let's keep strict logic for Yes/No as it is safer.
         token_match = exact_match
    else:
         token_match = any(pred_tok == rt for rt in refs_tok_list)

    # --- 3. BLEU (Normalized Tokens) ---
    refs_tok = refs_tok_list # Reuse
    
    if not pred_tok or not refs_tok:
        bleu = 0.0
    else:
        bleu = sentence_bleu(refs_tok, pred_tok, (1, 0, 0, 0), smoothing_function=SMOOTH)
    
    return {
        "exact_match": exact_match, # Strict
        "token_match": token_match, # Normalized/Soft
        "is_yesno": is_yesno_question,
        "bleu": bleu,
        "normalized_pred": " ".join(pred_tok),
        "normalized_refs": [" ".join(r) for r in refs_tok]
    }

# ================= HELPER FUNCTIONS =================

def find_adapter_dir(search_path: str) -> str:
    if not os.path.exists(search_path): return ""
    final = os.path.join(search_path, "final_adapter")
    if os.path.isdir(final): return final
    ckpts = sorted(glob.glob(os.path.join(search_path, "checkpoint-*")))
    if ckpts:
        ckpts.sort(key=lambda p: int(p.split("-")[-1]))
        return ckpts[-1]
    return search_path if os.path.isdir(search_path) else ""

@st.cache_resource
def load_med_model(base_model_id, adapter_path, use_4bit=True):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    ) if use_4bit else None

    state_txt = st.empty()
    state_txt.info(f"Loading base model: {base_model_id}...")
    
    base = LlavaForConditionalGeneration.from_pretrained(
        base_model_id, device_map="auto", torch_dtype=torch.float16,
        quantization_config=bnb_config, low_cpu_mem_usage=True
    )

    real_adapter = find_adapter_dir(adapter_path)
    if not real_adapter:
        st.error(f"Adapter not found in: {adapter_path}")
        return None, None

    state_txt.info(f"Loading adapter: {real_adapter}...")
    model = PeftModel.from_pretrained(base, real_adapter)
    model.eval()

    processor = LlavaProcessor.from_pretrained(base_model_id)
    state_txt.empty()
    return model, processor

def load_test_data(txt_path):
    rows = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 4:
                rows.append({
                    "image_id": parts[0].strip(),
                    "category": parts[1].strip(),
                    "question": parts[2].strip(),
                    "answer": parts[3].strip()
                })
    return pd.DataFrame(rows)

def build_prompt(question: str) -> str:
    return f"SYSTEM: {SYSTEM_PROMPT}\nUSER: <image>\nQuestion: {question}\nAnswer briefly with a short phrase.\nASSISTANT:"

def clean_pred(text: str) -> str:
    if not text: return ""
    t = text.strip().replace("\u200b", "")
    if "\n" in t: t = t.splitlines()[0]
    for bad in ["ASSISTANT:", "Assistant:", "assistant:"]:
        if t.startswith(bad): t = t[len(bad):].strip()
    return t

# ================= UI LOGIC =================

# ================= BATCH METRICS LOGIC =================

def extract_json_pairs(obj):
    """
    Parses various JSON formats (flat list, dict with 'samples', grouped by ID)
    Returns list of (ground_truth, predicted, category)
    """
    pairs = []
    
    def _get_cat(item):
        return item.get("category", "unknown")

    # Case 1: List of objects
    if isinstance(obj, list):
        for s in obj:
            if isinstance(s, dict):
                pairs.append((s.get("ground_truth", ""), s.get("predicted", ""), _get_cat(s)))
        return pairs

    if isinstance(obj, dict):
        # Case 2: {"samples": [...]}
        if "samples" in obj and isinstance(obj["samples"], list):
            for s in obj["samples"]:
                # specific format with qas list
                if "qas" in s: 
                    for qa in s["qas"]:
                        pairs.append((qa.get("ground_truth", ""), qa.get("predicted", ""), _get_cat(qa)))
                else:
                    pairs.append((s.get("ground_truth", ""), s.get("predicted", ""), _get_cat(s)))
            return pairs
            
        # Case 3: Dict grouped by ID {id: [qa, ...]}
        for _, qa_list in obj.items():
            if isinstance(qa_list, list):
                for qa in qa_list:
                    if isinstance(qa, dict):
                        pairs.append((qa.get("ground_truth", ""), qa.get("predicted", ""), _get_cat(qa)))
                        
    return pairs

def compute_batch_stats(pairs):
    stats = []
    for gt, pred, cat in pairs:
        # Reuse single score logic
        s = calculate_single_score(pred, gt)
        stats.append({
            "category": cat,
            "acc_strict": 1 if s["exact_match"] else 0,
            "acc_norm": 1 if (s["exact_match"] if s["is_yesno"] else s["token_match"]) else 0,
            "bleu": s["bleu"],
            "is_yesno": s["is_yesno"]
        })
    return pd.DataFrame(stats)

# ================= UI LOGIC =================

st.title("🏥 Medical VQA Assistant")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    app_mode = st.radio("Mode", ["Single Image Inference", "Batch Metrics Dashboard"], index=0)
    st.divider()

    if app_mode == "Single Image Inference":
        st.info("Upload any image. If it's from the VQA-Med Test Set, we'll auto-suggest valid questions!")
        st.divider()
        model_id = st.text_input("Base Model", value=DEFAULT_MODEL_ID)
        adapter_dir = st.text_input("Adapter Path", value=DEFAULT_ADAPTER_DIR)
        use_4bit = st.checkbox("Use 4-bit (Save VRAM)", value=True)
        
        if st.button("Reload Model"):
            st.cache_resource.clear()
            st.session_state.model_loaded = False
            st.rerun()
            
        st.divider()
        if torch.cuda.is_available():
            st.success(f"⚡ GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.error("⚠️ CPU Mode (Slow)")

# ================= APP: SINGLE INFERENCE =================
if app_mode == "Single Image Inference":
    # --- Load Model ---
    try:
        model, processor = load_med_model(model_id, adapter_dir, use_4bit)
        if not model: st.stop()
    except Exception as e:
        st.error(f"Critial Error: {e}")
        st.stop()

    # --- Load Test Data (Background) ---
    try:
        df_test = load_test_data(TEST_TXT_PATH)
        valid_ids = set(df_test["image_id"].values)
    except Exception:
        df_test = None
        valid_ids = set()

    # --- Main Layout ---
    col_img, col_qa = st.columns([1, 1.2])

    with col_img:
        st.subheader("1. Upload Image")
        uploaded_file = st.file_uploader("Drop an image here...", type=["jpg", "png", "jpeg"])
        
        image = None
        file_id = None
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_column_width=True)
            
            # Smart Detection: Extract ID from filename (e.g. "synpic1234.jpg" -> "synpic1234")
            file_id = os.path.splitext(uploaded_file.name)[0]
            if file_id in valid_ids:
                st.success(f"✅ Recognized as Test Image: `{file_id}`")
            else:
                st.info("Custom Image (Not in Test Set DB)")

    with col_qa:
        st.subheader("2. Select Question")
        
        question = ""
        ground_truth = None
        
        # Logic: If recognized, show dropdown. Else, text input.
        if file_id and file_id in valid_ids and df_test is not None:
            # Filter questions
            subset = df_test[df_test["image_id"] == file_id]
            
            # Selection Mode
            q_mode = st.radio("Question Mode", ["Select Valid Question", "Custom Question"], horizontal=True)
            
            if q_mode == "Select Valid Question":
                q_map = {row["question"]: row["answer"] for _, row in subset.iterrows()}
                question = st.selectbox("Available Questions:", list(q_map.keys()))
                ground_truth = q_map.get(question)
                
                if not question: st.warning("No questions found for this ID.")
                
                # Show Category info
                cat = subset[subset['question'] == question]['category'].values[0] if question else "Unknown"
                st.caption(f"Category: {cat}")
                
            else:
                question = st.text_input("Type your question:", placeholder="e.g. What organ is this?")
                
        else:
            # Fallback for custom images
            st.warning("Model trained on: Organ, Modality, Plane, Abnormality.")
            question = st.text_input("Type your question:", placeholder="e.g. What plane is this ct scan?")

        # --- Prediction ---
        st.divider()
        run_btn = st.button("🔍 Predict Answer", type="primary", disabled=(image is None or not question))
        
        if run_btn:
            with st.spinner("Analyzing..."):
                inputs = processor(text=build_prompt(question), images=image, return_tensors="pt").to(model.device)
                gen = model.generate(**inputs, max_new_tokens=40)
                pred = processor.tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                pred_clean = clean_pred(pred)
            
            # Display Results
            if ground_truth:
                # Metric Calculation
                scores = calculate_single_score(pred_clean, ground_truth)
                
                c1, c2 = st.columns(2)
                c1.markdown("### 🤖 Prediction")
                c1.success(f"**{pred_clean}**")
                
                c2.markdown("### ✅ Ground Truth")
                c2.info(f"**{ground_truth}**")
                
                st.divider()
                
                # Metric UI
                m1, m2 = st.columns(2)
                
                # Use Token Match (Normalized) as the primary "Accuracy" per user request
                lbl = "Accuracy (Strict)" if scores["is_yesno"] else "Accuracy (Normalized)"
                val = scores["exact_match"] if scores["is_yesno"] else scores["token_match"]
                
                if val:
                    m1.metric(lbl, "Correct", delta="Match", delta_color="normal")
                    st.balloons()
                else:
                    m1.metric(lbl, "Incorrect", delta="Mismatch", delta_color="inverse")
                
                m2.metric("BLEU-1 Score", f"{scores['bleu']:.4f}")
                
                with st.expander("Show Normalization Details"):
                    st.write(f"Normalized Prediction: `{scores['normalized_pred']}`")
                    st.write(f"Normalized Ref(s): `{scores['normalized_refs']}`")
                    st.caption("Normalization used: Lowercase, Remove Punctuation, Remove Stopwords, Stemming (Snowball).")

            else:
                st.markdown("### 🤖 Prediction")
                st.success(f"**{pred_clean}**")

# ================= APP: BATCH METRICS DASHBOARD =================
elif app_mode == "Batch Metrics Dashboard":
    st.header("📊 Batch Evaluation Dashboard")
    st.write("Upload a `preds.json` file to calculate aggregate metrics for the entire test set.")
    
    uploaded_json = st.file_uploader("Upload Predictions JSON", type=["json"])
    
    if uploaded_json:
        import json
        try:
            data = json.load(uploaded_json)
            pairs = extract_json_pairs(data)
            
            if not pairs:
                st.error("Could not extract any (ground_truth, predicted) pairs from this JSON.")
            else:
                st.success(f"Loaded {len(pairs)} predictions.")
                
                # Compute Stats
                df_stats = compute_batch_stats(pairs)
                
                # Overall Metrics
                acc_strict = df_stats["acc_strict"].mean() * 100
                acc_norm = df_stats["acc_norm"].mean() * 100
                bleu_avg = df_stats["bleu"].mean()
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Overall Accuracy (Strict)", f"{acc_strict:.2f}%", help="Exact string match")
                m2.metric("Overall Accuracy (Normalized)", f"{acc_norm:.2f}%", help="Token match (handles formatting/stemming)")
                m3.metric("Avg BLEU-1", f"{bleu_avg:.4f}")
                
                st.divider()
                
                # Category Breakdown
                st.subheader("Category Performance")
                cat_grp = df_stats.groupby("category")[["acc_norm", "bleu"]].mean().reset_index()
                cat_grp["acc_norm"] = cat_grp["acc_norm"] * 100
                
                c_chart, c_table = st.columns([2, 1])
                
                with c_chart:
                    st.bar_chart(cat_grp.set_index("category")["acc_norm"], color="#4CAF50")
                    st.caption("Normalized Accuracy per Category")
                    
                with c_table:
                    st.dataframe(cat_grp.style.format({"acc_norm": "{:.2f}%", "bleu": "{:.4f}"}))
                
                # Detailed Errors (Optional)
                with st.expander("Show All Data"):
                    st.dataframe(pd.DataFrame(pairs, columns=["Ground Truth", "Prediction", "Category"]))
                    
        except Exception as e:
            st.error(f"Error parsing JSON: {e}")

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if not get_script_run_ctx():
            print("\n⚠️ WARNING: Run with 'streamlit run app.py'")
    except ImportError: pass
