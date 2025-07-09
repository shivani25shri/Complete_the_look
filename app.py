# app.py
import os
import time
import base64
import numpy as np
import torch
import requests
import streamlit as st
from PIL import Image
from pymongo import MongoClient
from google import genai
from google.genai import types
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel

# ─── 1) CONFIGURE CLIENTS ────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyBO_6Afs_Ub45w6qoWDCjYDxVFA-siZR8k"
client         = genai.Client(api_key=GEMINI_API_KEY)
GEN_MODEL      = "gemini-2.5-flash-preview-04-17"

mongo        = MongoClient("mongodb+srv://shivani25shri10:bn7Reynw8ymF2ytC@cluster0.c2asd9y.mongodb.net/")
collection   = mongo["fashionista_"]["catalog_data"]

# Local e5 embedder (384-dim)
TOKENIZER_PATH = "./e5-small-v2"
tokenizer_e5   = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
model_e5       = AutoModel.from_pretrained(TOKENIZER_PATH, local_files_only=True).eval()

# CLIP for gender detection
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()

# ─── 2) UTILITIES ────────────────────────────────────────────────────────────
def infer_category_from_filename(path: str) -> str:
    fn = os.path.basename(path).lower()
    bottoms = ("pant", "trouser", "short", "skirt", "dress")
    return "bottom wear" if any(tok in fn for tok in bottoms) else "top wear"

def detect_gender(image_path: str) -> str:
    labels = ["men's clothing", "women's clothing", "unisex clothing"]
    img = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=labels, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0]
    return labels[int(probs.argmax())]

def load_uri(path: str) -> str:
    raw = open(path, "rb").read()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()

def embed_local(text: str) -> np.ndarray:
    inp = tokenizer_e5(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        hs   = model_e5(**inp).last_hidden_state
        mask = inp["attention_mask"].unsqueeze(-1).expand(hs.size())
        summed = (hs * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        vec    = (summed / counts)[0]
    return vec.cpu().numpy()

def retrieve_similar(emb: np.ndarray, gender: str, k: int = 5):
    qn = emb / np.linalg.norm(emb)
    docs = list(collection.find({"gender": gender}, {"text":1, "image_url":1, "embedding":1}))
    if not docs:
        docs = list(collection.find({}, {"text":1, "image_url":1, "embedding":1}))
    scored = []
    for d in docs:
        e  = np.array(d["embedding"], dtype=np.float32)
        en = e / np.linalg.norm(e)
        scored.append((float(np.dot(qn, en)), d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]

# ─── 3) GEMINI PROMPT ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a world-class fashion stylist. You will always choose complementary pieces from the *other* category first:

- If Category is “bottom wear” (jeans, skirt, trousers, shorts), you MUST suggest top wear first, then shoes and accessories.
- If Category is “top wear” (blouse, jacket, sweater), you MUST suggest bottom wear first, then shoes and accessories.

Answer in 1–2 sentences.
""".strip()

def generate_stylist_description(image_path: str, category: str, gender: str) -> str:
    uri = load_uri(image_path)
    user_prompt = (
        f"Here is the image: {uri}\n"
        f"Category: {category}\n"
        f"Gender: {gender}\n\n"
        "Please describe in 1–2 sentences what items and accessories would complete the look."
    )
    sys_part = types.Part.from_text(text=SYSTEM_PROMPT)
    usr_part = types.Part.from_text(text=user_prompt)
    cfg      = types.GenerateContentConfig(response_mime_type="text/plain")
    stream   = client.models.generate_content_stream(
        model=GEN_MODEL,
        contents=[
            types.Content(role="model", parts=[sys_part]),
            types.Content(role="user",  parts=[usr_part])
        ],
        config=cfg
    )
    return "".join(chunk.text for chunk in stream).strip()

# ─── 4) STREAMLIT UI ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Complete the Look", layout="wide")
st.title("👗 Complete the Look")

# Inject custom CSS
st.markdown("""
    <style>
    .stImage > img {
        border-radius: 10px;
    }
    .caption {
        font-size: 0.7rem !important;
        color: #6c6c6c;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    h1, h2, h3, h4, h5, h6 {
        font-size: 1.2rem !important;
    }
    .stButton button {
        padding: 0.4rem 0.8rem;
        font-size: 0.9rem;
    }
    .stSelectbox label, .stFileUploader label {
        font-size: 0.9rem;
    }
    .stMarkdown p {
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload your garment image", type=["jpg","jpeg","png"])
category_override = st.selectbox("Category (override auto-infer)", ["Auto", "top wear", "bottom wear"])

if uploaded:
    # Save to temp file
    tmp_dir = "tmp_uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    file_path = os.path.join(tmp_dir, uploaded.name)
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())
    

    st.image(file_path, use_column_width=300)
    st.markdown('<div class="caption">Your Input</div>', unsafe_allow_html=True)

    if st.button("Show Recommendations"):
        # 1) Category
        if category_override == "Auto":
            category = infer_category_from_filename(file_path)
            st.info(f"Auto-inferred category: **{category}**")
        else:
            category = category_override

        # 2) Gender
        gender = detect_gender(file_path)
        st.write(f"**Target Gender:** {gender}")

        # 3) Gemini description
        advice = generate_stylist_description(file_path, category, gender)
        st.subheader("Styling Advice")
        st.write(advice)

        # 4) Embedding & retrieval
        emb   = embed_local(advice)
        recs  = retrieve_similar(emb, gender, k=5)

        # 5) Display as a single horizontal row
        st.subheader("Recommended Pairings")
        cols = st.columns(5)
        for col, doc in zip(cols, recs):
            col.image(doc["image_url"], use_column_width=True)
            col.caption(doc["text"])
