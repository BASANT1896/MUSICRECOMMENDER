# recommend.py
import pandas as pd
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)

def load_data(df_file, sim_file):
    try:
        import joblib
        df = joblib.load(df_file)
        cosine_sim = joblib.load(sim_file)
        logging.info("✅ Data loaded successfully from upload.")
        return df, cosine_sim
    except Exception as e:
        logging.error("❌ Failed to load uploaded files: %s", str(e))
        raise e

def recommend_songs(df, cosine_sim, song_name, top_n=5):
    logging.info("🎵 Recommending songs for: '%s'", song_name)
    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        logging.warning("⚠️ Song not found in dataset.")
        return None
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]
    result_df = df[['artist', 'song']].iloc[song_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."
    return result_df
