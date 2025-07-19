# recommend.py

import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')

@st.cache_data(show_spinner=False)
def load_df():
    df = pd.read_csv("spotify_millsongdata.csv").sample(5000).drop(columns=["link"], errors="ignore").reset_index(drop=True)
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        text = re.sub(r"[^a-zA-Z\s]", "", str(text))
        tokens = word_tokenize(text.lower())
        return " ".join([word for word in tokens if word not in stop_words])

    df["cleaned_text"] = df["text"].apply(preprocess)
    return df

@st.cache_resource(show_spinner=False)
def compute_similarity(df):
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend_songs(df, cosine_sim, song_name, top_n=5):
    if song_name not in df["song"].values:
        return None

    idx = df[df["song"] == song_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended = df.iloc[[i[0] for i in sim_scores]][["artist", "song"]]
    return recommended
