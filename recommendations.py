import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset():
    df = pd.read_csv("skincare_ingredients.csv")
    df["ingredients_clean"] = (
        df["ingredients"]
        .str.lower()
        .str.replace(r"[^\w\s,]", "", regex=True)
        .str.replace(",", " ")
    )
    return df

def build_similarity(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["ingredients_clean"])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

def hybrid_recommend(product_name, df, cosine_sim, alpha=0.7, k=5):
    try:
        idx = df[df["name"].str.lower() == product_name.lower()].index[0]
    except:
        return None

    target_skin = df.loc[idx, ["Combination","Dry","Normal","Oily","Sensitive"]]
    skin_matrix = df[["Combination","Dry","Normal","Oily","Sensitive"]].values
    skin_sim = (skin_matrix == target_skin.values).sum(axis=1) / 5

    hybrid_score = (alpha * cosine_sim[idx]) + ((1 - alpha) * skin_sim)

    df["Hybrid_Score"] = hybrid_score
    df_sorted = df.sort_values("Hybrid_Score", ascending=False)
    df_sorted = df_sorted[df_sorted.index != idx]

    return df_sorted[["name","brand","price","Hybrid_Score"]].head(k)
