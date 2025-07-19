# recommend.py
import os
import joblib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)

df = None  # Global dataframe

def load_df():
    global df
    if df is not None:
        return df  # already loaded

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        df_path = os.path.join(base_dir, 'df_cleaned.pkl')
        df = joblib.load(df_path)

        if 'similarities' not in df.columns:
            raise ValueError("Missing 'similarities' column in df_cleaned.pkl")

        logging.info("✅ df_cleaned.pkl loaded successfully.")
        return df

    except Exception as e:
        logging.error("❌ Failed to load df_cleaned.pkl: %s", str(e))
        return None


def recommend_songs(song_name, top_n=5):
    df_local = load_df()
    if df_local is None:
        logging.error("❌ DataFrame is not loaded.")
        return None

    logging.info("🎵 Recommending songs for: '%s'", song_name)
    idx = df_local[df_local['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        logging.warning("⚠️ Song not found in dataset.")
        return None

    idx = idx[0]
    similarities = df_local.at[idx, 'similarities']
    sim_scores = list(enumerate(similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]

    result_df = df_local[['artist', 'song']].iloc[song_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."
    return result_df
