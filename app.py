import pandas as pd
import streamlit as st
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book Recommendation System")
st.caption("TF-IDF + Cosine Similarity (Dataset Buku Indonesia)")

# =========================
# LOAD DATA (SAFE PATH)
# =========================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "books_indonesia.csv")

    df = pd.read_csv(file_path)

    # safety columns
    if "title" not in df.columns:
        df["title"] = ""
    if "description" not in df.columns:
        df["description"] = ""

    df = df[["title", "description"]].fillna("")
    return df

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("ℹ️ Info")
st.sidebar.info("TF-IDF + Cosine Similarity untuk rekomendasi buku Indonesia")

top_n = st.sidebar.slider("Jumlah rekomendasi", 1, 10, 5)

# =========================
# PREPROCESSING
# =========================
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["text"] = (df["title"] + " " + df["description"]).apply(preprocess)

# =========================
# TF-IDF MODEL (CACHE)
# =========================
@st.cache_resource
def build_model(texts):
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )
    matrix = tfidf.fit_transform(texts)
    return tfidf, matrix

tfidf, tfidf_matrix = build_model(df["text"])

# =========================
# RECOMMENDATION ENGINE
# =========================
def recommend_books(query, top_n):
    query = preprocess(query)

    if not query.strip():
        return []

    vec = tfidf.transform([query])
    sim = cosine_similarity(vec, tfidf_matrix)[0]

    top_idx = sim.argsort()[::-1][:top_n]

    results = []
    for i in top_idx:
        results.append({
            "title": df.iloc[i]["title"],
            "description": df.iloc[i]["description"],
            "score": float(sim[i])
        })

    return results

# =========================
# UI
# =========================
user_input = st.text_area("Masukkan deskripsi buku:")

if st.button("🔍 Cari Rekomendasi"):

    if not user_input.strip():
        st.warning("Input tidak boleh kosong!")
    elif df.empty:
        st.error("Dataset kosong atau tidak berhasil dimuat.")
    else:
        results = recommend_books(user_input, top_n)

        if not results:
            st.info("Tidak ada hasil rekomendasi.")
        else:
            for r in results:
                st.markdown(
                    f"""
                    <div style="
                        padding:15px;
                        border-radius:12px;
                        border:1px solid #ddd;
                        margin-bottom:12px;
                        background:linear-gradient(90deg,#111827,#1f2937);
                        color:white;
                    ">
                        <h4>📖 {r['title']}</h4>
                        <p>{r['description'][:250]}...</p>
                        <p><b>Similarity:</b> {round(r['score'],4)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )