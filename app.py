import streamlit as st
import pandas as pd
import altair as alt

from recommendations import (
    load_dataset,
    build_similarity,
    hybrid_recommend_by_index
)
from evaluation import precision_at_k_single


# SESSION STATE
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# LOAD CSS
with open("styles/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["Rekomendasi Produk", "Dataset & Insight"]
)

# HEADER
st.markdown("<div class='big-title'>Sistem Rekomendasi Skincare</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Rekomendasi berbasis ingredients & jenis kulit</div>",
    unsafe_allow_html=True
)

# LOAD DATA
df = load_dataset()
cosine_sim = build_similarity(df)

brand_count = df["brand"].nunique()
label_count = df["Label"].nunique()


# MENU 1 — REKOMENDASI
if menu == "Rekomendasi Produk":

    st.markdown(
        f"""
        <div class='dataset-info'>
            Dataset berisi <b>{len(df)}</b> produk dari <b>{brand_count}</b> brand
            dengan <b>{label_count}</b> kategori skincare.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Input Produk")

    # SEARCH
    search = st.text_input("Cari Produk", placeholder="misal: toner, moisturizer")

    if search:
        filtered_df = df[df["name"].str.contains(search, case=False, na=False)]
    else:
        filtered_df = df

    # SELECT PRODUCT BY INDEX (AMAN)
    selected_idx = st.selectbox(
        "Pilih Produk",
        filtered_df.index,
        format_func=lambda i: filtered_df.loc[i, "name"]
    )

    skin_type = st.selectbox(
        "Jenis Kulit",
        ["Normal", "Dry", "Oily", "Combination", "Sensitive"]
    )

    alpha = st.slider("Nilai Alpha", 0.0, 1.0, 0.7, 0.1)
    k = st.number_input("Jumlah Rekomendasi (Top-K)", 1, 20, 5)

    if st.button("Dapatkan Rekomendasi"):
        st.session_state.submitted = True

    # HASIL 
    if st.session_state.submitted:
        st.subheader("Hasil Rekomendasi")

        results = hybrid_recommend_by_index(
            selected_idx,
            df,
            cosine_sim,
            alpha,
            k,
            user_skin=skin_type
        )

        if results is None or results.empty:
            st.warning("Tidak ada rekomendasi.")
        else:
            results["Hybrid_Score"] = results["Hybrid_Score"].round(3)
            st.dataframe(results, use_container_width=True)

            # Precision (evaluasi individual)
            query_name = df.loc[selected_idx, "name"]

            precision = precision_at_k_single(
                df,
                query_name,
                results["name"].tolist(),
                k
            )

            st.metric("Precision@K (Evaluasi Individual)", round(precision, 3))

# MENU 2 — DATASET
elif menu == "Dataset & Insight":

    st.subheader("Dataset & Insight")

    st.write(
        f"""
        - Total Produk: {len(df)}  
        - Total Brand: {brand_count}  
        - Kategori Skincare: {label_count}
        """
    )

    ingredients_exploded = df["ingredients"].str.split(",").explode()
    top20 = ingredients_exploded.value_counts().head(20).reset_index()
    top20.columns = ["ingredient", "count"]

    chart = (
        alt.Chart(top20)
        .mark_bar()
        .encode(
            x="count:Q",
            y=alt.Y("ingredient:N", sort="-x")
        )
        .properties(
            height=450,
            title="Top 20 Ingredients Paling Umum"
        )
    )

    st.altair_chart(chart, use_container_width=True)
