# recommend.py

import pandas as pd
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

logging.basicConfig(level=logging.INFO)
df_path = "df_cleaned.pkl"

@st.cache_data(show_spinner="📦 Loading data...")
def load_df():
    try:
        logging.info(f"📄 Loading df from: {df_path}")
        df = joblib.load(df_path)
        if 'cleaned_text' not in df.columns:
            raise ValueError("Missing 'cleaned_text' column in df_cleaned.pkl")
        return df
    except Exception as e:
        logging.error("❌ Failed to load df_cleaned.pkl: %s", str(e))
        raise e

@st.cache_resource(show_spinner="🔄 Computing similarity...")
def compute_similarity(df):
    logging.info("📐 Computing TF-IDF and cosine similarity...")
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

def recommend_songs(df, cosine_sim, title, top_n=5):
    title = title.lower()
    indices = df[df['title'].str.lower() == title].index

    if len(indices) == 0:
        return []

    idx = indices[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in similarity_scores]

    return df.iloc[song_indices][['title', 'artist']].to_dict(orient='records')

