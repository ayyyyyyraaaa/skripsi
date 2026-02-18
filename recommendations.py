import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# UTIL: NORMALIZE NAME
def normalize_name(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# LOAD DATASET
def load_dataset():
    df = pd.read_csv("skincare_ingredients.csv")
    df.columns = [c.strip() for c in df.columns]

    # Simpan nama original & normalized
    df["name_norm"] = df["name"].apply(normalize_name)

    # Clean ingredients
    df["ingredients_clean"] = (
        df["ingredients"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^\w\s,]", "", regex=True)
        .str.replace(",", " ")
    )

    if "Label" not in df.columns and "type" in df.columns:
        df["Label"] = df["type"]

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    else:
        df["price"] = 0

    return df

# BUILD SIMILARITY
def build_similarity(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["ingredients_clean"])
    return cosine_similarity(tfidf_matrix)

# HYBRID RECOMMENDER
def hybrid_recommend_by_index(idx, df, cosine_matrix, alpha=0.7, k=5, user_skin=None):
    ingredient_sim = cosine_matrix[idx]

    if user_skin and user_skin in df.columns:
        skin_weight = df[user_skin].values.astype(float)
    else:
        skin_weight = 0

    final_score = (alpha * ingredient_sim) + ((1 - alpha) * skin_weight)

    scores = list(enumerate(final_score))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_idx = [i for i, _ in scores if i != idx][:k]

    rec = df.iloc[top_idx].copy()
    rec["Hybrid_Score"] = final_score[top_idx]

    return rec[["name", "brand", "Label", "price", "Hybrid_Score"]].reset_index(drop=True)
