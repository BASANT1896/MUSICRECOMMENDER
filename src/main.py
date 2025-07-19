# main.py

import streamlit as st
from recommend import load_df, compute_similarity, recommend_songs

# Set custom Streamlit page config
st.set_page_config(
    page_title="Music Recommender 🎵",
    page_icon="🎧",
    layout="centered"
)

st.title("🎶 Instant Music Recommender")

# Load data and compute similarity on demand
df = load_df()
cosine_sim = compute_similarity(df)

# Song selection
song_list = sorted(df['song'].dropna().unique())
selected_song = st.selectbox("🎵 Select a song:", song_list)

if st.button("🚀 Recommend Similar Songs"):
    with st.spinner("Finding similar songs..."):
        recommendations = recommend_songs(df, cosine_sim, selected_song)

        if recommendations is None:
            st.warning("Sorry, song not found.")
        elif recommendations.empty:
            st.warning("No similar songs found.")
        else:
            st.success("Top similar songs:")
            st.table(recommendations)
