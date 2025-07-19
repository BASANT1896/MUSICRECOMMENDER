#main.py
import streamlit as st
from recommend import recommend_songs, df

# Set Streamlit page config
st.set_page_config(
    page_title="Music Recommender 🎵",
    page_icon="🎧",
    layout="centered"
)

st.title("🎶 Instant Music Recommender")

# Check if DataFrame was loaded successfully
if df is None:
    st.error("❌ Failed to load dataset. Please check your df_cleaned.pkl in the `src/` folder.")
else:
    song_list = sorted(df['song'].dropna().unique())
    selected_song = st.selectbox("🎵 Select a song:", song_list)

    if st.button("🚀 Recommend Similar Songs"):
        with st.spinner("Finding similar songs..."):
            recommendations = recommend_songs(selected_song)
            if recommendations is None:
                st.warning("⚠️ Song not found.")
            else:
                st.success("✅ Top similar songs:")
                st.table(recommendations)
