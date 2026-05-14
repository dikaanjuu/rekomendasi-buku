import pandas as pd
import streamlit as st
import re

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
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("books_indonesia.csv")

    # pastikan kolom aman
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
st.sidebar.info(
    "Sistem rekomendasi buku berbasis TF-IDF dan Cosine Similarity "
    "menggunakan dataset buku Indonesia."
)

top_n = st.sidebar.slider("Jumlah rekomendasi", 1, 10, 5)

# =========================
# PREPROCESSING (SIMPLER & ROBUST)
# =========================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df["clean_text"] = df["title"] + " " + df["description"]
df["clean_text"] = df["clean_text"].apply(preprocess)

# =========================
# TF-IDF (IMPORTANT: CACHE MODEL)
# =========================
@st.cache_resource
def train_tfidf(data):
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )
    matrix = tfidf.fit_transform(data)
    return tfidf, matrix

tfidf, tfidf_matrix = train_tfidf(df["clean_text"])

# =========================
# RECOMMENDATION FUNCTION
# =========================
def recommend_books(input_text, top_n):
    input_text = preprocess(input_text)
    input_vector = tfidf.transform([input_text])

    similarity = cosine_similarity(input_vector, tfidf_matrix)

    scores = list(enumerate(similarity[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []

    for i, score in scores[:top_n]:
        results.append({
            "title": df.iloc[i]["title"],
            "description": df.iloc[i]["description"],
            "score": float(score)
        })

    return results

# =========================
# UI INPUT
# =========================
user_input = st.text_area(
    "Masukkan deskripsi buku:",
    placeholder="Contoh: kisah cinta remaja di bandung, atau dunia fantasy magic kingdom"
)

search_btn = st.button("🔍 Cari Rekomendasi")

# =========================
# OUTPUT
# =========================
if search_btn:

    if user_input.strip() == "":
        st.warning("Input tidak boleh kosong!")
    else:
        with st.spinner("Mencari rekomendasi..."):
            results = recommend_books(user_input, top_n)

        st.success("Rekomendasi ditemukan!")

        for r in results:
            st.markdown(
                f"""
                <div style="
                    padding:15px;
                    border-radius:12px;
                    border:1px solid #ddd;
                    margin-bottom:12px;
                    background:linear-gradient(90deg,#f9f9f9,#f1f5ff);
                ">
                    <h4>📖 {r['title']}</h4>
                    <p>{r['description'][:300]}...</p>
                    <p><b>Similarity:</b> {round(r['score'],4)}</p>
                </div>
                """,
                unsafe_allow_html=True
            )