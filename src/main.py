# app.py
import streamlit as st
from recommend import load_data, recommend_songs

# Set Streamlit page config
st.set_page_config(
    page_title="Music Recommender 🎵",
    page_icon="🎧",
    layout="centered"
)

st.title("🎶 Instant Music Recommender")

# File upload section
st.sidebar.header("🔽 Upload Required Files")
df_file = st.sidebar.file_uploader("Upload `df_cleaned.pkl`", type="pkl")
sim_file = st.sidebar.file_uploader("Upload `cosine_sim.pkl`", type="pkl")

# Load files
if df_file and sim_file:
    with st.spinner("Loading uploaded files..."):
        df, cosine_sim = load_data(df_file, sim_file)

    song_list = sorted(df['song'].dropna().unique())
    selected_song = st.selectbox("🎵 Select a song:", song_list)

    if st.button("🚀 Recommend Similar Songs"):
        with st.spinner("Finding similar songs..."):
            recommendations = recommend_songs(df, cosine_sim, selected_song)
            if recommendations is None:
                st.warning("⚠️ Song not found.")
            else:
                st.success("✅ Top similar songs:")
                st.table(recommendations)
else:
    st.info("📂 Please upload both `.pkl` files from your local system.")
