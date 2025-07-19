#recommend.py
import os
import joblib
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)

# Use absolute path to df_cleaned.pkl inside src/
df_path = Path(__file__).parent / "df_cleaned.pkl"

try:
    logging.info(f"📄 Loading from: {df_path}")
    df = joblib.load(df_path)
       
    cosine_sim = joblib.load("cosine_sim.pkl")

    logging.info("✅ df_cleaned.pkl loaded successfully.")
except Exception as e:
    logging.error("❌ Failed to load df_cleaned.pkl: %s", str(e))
    df = None
    raise e  # <- important so you know it fails during dev


def recommend_songs(song_name, top_n=5):
    logging.info("🎵 Recommending songs for: '%s'", song_name)
    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        logging.warning("⚠️ Song not found in dataset.")
        return None
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]
    logging.info("✅ Top %d recommendations ready.", top_n)
    # Create DataFrame with clean serial numbers starting from 1
    result_df = df[['artist', 'song']].iloc[song_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1  # Start from 1 instead of 0
    result_df.index.name = "S.No."

    return result_df
