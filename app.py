import streamlit as st
import pandas as pd
import altair as alt

from recommendations import load_dataset, build_similarity, hybrid_recommend
from evaluation import precision_at_k, recall_at_k, mean_average_precision

# ========== LOAD CSS ==========
with open("styles/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ========== HEADER SECTION ==========
st.markdown("<div class='big-title'>Skincare Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A content-based filtering system for skincare product recommendations</div>", unsafe_allow_html=True)

# ========== LOAD DATA ==========
df = load_dataset()
cosine_sim = build_similarity(df)

# ========== INPUT FORM ==========
st.markdown("<div class='section-title' style='text-align:center; font-size:28px; font-weight:700;'>Input Product Details</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='box'>", unsafe_allow_html=True)

    selected = st.selectbox("Pilih Produk", df["name"].tolist())
    alpha = st.slider("Nilai Alpha (0 = Skin, 1 = Ingredients)", 0.0, 1.0, 0.7, 0.1)
    k = st.number_input("Jumlah Rekomendasi", 1, 20, 5)

    submit = st.button("Get Recommendations")

    st.markdown("</div>", unsafe_allow_html=True)

# ========== SHOW RESULT ==========
if submit:
    st.markdown("<h3 style='text-align:center; margin-top:20px;'>Hasil Rekomendasi</h3>", unsafe_allow_html=True)
    
    hasil = hybrid_recommend(selected, df, cosine_sim, alpha, k)
    st.table(hasil)

    # ========== INGREDIENT DISTRIBUTION ==========
    ingredients_exploded = df["ingredients"].str.split(",").explode()
    top20 = ingredients_exploded.value_counts().head(20)
    chart_data = top20.reset_index()
    chart_data.columns = ["ingredient","count"]

    st.markdown("<h3 style='text-align:center; margin-top:40px;'>Ingredient Distribution</h3>", unsafe_allow_html=True)
    chart = alt.Chart(chart_data).mark_bar().encode(
        x="count:Q",
        y=alt.Y("ingredient:N", sort="-x")
    ).properties(width=700, height=400)
    st.altair_chart(chart)

    # ========== EVALUATION ==========
    st.markdown("<h3 style='text-align:center; margin-top:40px;'>Evaluation Result</h3>", unsafe_allow_html=True)

    # contoh ground truth dummy
    ground_truth = ["EGF Serum","EGF Day Serum"]

    pred_list = hasil["name"].tolist()

    st.write({
        "Precision@k": precision_at_k(ground_truth, pred_list, k),
        "Recall@k": recall_at_k(ground_truth, pred_list, k),
        "MAP": mean_average_precision(ground_truth, pred_list, k)
    })
