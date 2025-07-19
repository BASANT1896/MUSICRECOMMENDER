# recommend.py
import pandas as pd
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO)

df_path = "df_cleaned.pkl"

# Load the cleaned DataFrame
try:
    logging.info(f"📄 Loading df from: {df_path}")
    df = joblib.load(df_path)

    if 'cleaned_text' not in df.columns:
        raise ValueError("Missing 'cleaned_text' column in df_cleaned.pkl")

    logging.info("✅ df_cleaned.pkl loaded successfully.")

except Exception as e:
    logging.error("❌ Failed to load df_cleaned.pkl: %s", str(e))
    df = None
    raise e


# Compute cosine similarity at runtime (workaround)
@st.cache_resource(show_spinner="🔄 Computing similarity...")
def compute_similarity(cleaned_texts):
    logging.info("📐 Computing TF-IDF and cosine similarity matrix...")
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(cleaned_texts)
    cosine_sim = cosine_similarity(tfidf_matrix)
    logging.info("✅ Cosine similarity computed.")
    return cosine_sim

# Actually generate it
cosine_sim = compute_similarity(df['cleaned_text'])

# 🎯 Recommendation function using cosine_sim
def recommend_songs(title, top_n=5):
    if df is None:
        return []

    title = title.lower()
    indices = df[df['title'].str.lower() == title].index

    if len(indices) == 0:
        return []

    idx = indices[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in similarity_scores]

    return df.iloc[song_indices][['title', 'artist']].to_dict(orient='records')
