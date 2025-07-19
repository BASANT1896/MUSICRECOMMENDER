#preprocess.py
import pandas as pd
import re
import nltk
import joblib
import logging
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load and sample dataset
csv_path = "spotify_millsongdata.csv"
try:
    df = pd.read_csv(csv_path).sample(10000)
    logging.info("✅ Dataset loaded and sampled: %d rows", len(df))
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

# Drop 'link' column if exists
df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

# Preprocess lyrics text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

logging.info("🧹 Cleaning lyrics text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
logging.info("✅ Text cleaning complete.")

# Vectorization
logging.info("🔠 Vectorizing with TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

# Cosine similarity
logging.info("📐 Calculating cosine similarity (this may take time)...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("✅ Cosine similarity computed.")

# Embed similarity into DataFrame
df['similarities'] = list(cosine_sim)
logging.info("🧬 Similarity vectors embedded into DataFrame.")

# Save final DataFrame
filename = 'df_cleaned.pkl'
joblib.dump(df, filename)
size = os.path.getsize(filename) / (1024 * 1024)
logging.info(f"💾 Saved {filename} ({size:.2f} MB)")

logging.info("✅ Preprocessing complete.")

