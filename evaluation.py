def precision_at_k_single(df, query_name, preds, k):
    """
    Ingredient-based Precision@K (Proxy Evaluation)

    Produk dianggap relevan apabila memiliki rasio
    kemiripan ingredients >= 0.3 terhadap produk acuan.
    """

    query_row = df[df["name"] == query_name]
    if query_row.empty:
        return 0

    query_ings = set(
        query_row.iloc[0]["ingredients_clean"].split()
    )

    if len(query_ings) == 0:
        return 0

    match = 0
    for rec in preds[:k]:
        rec_row = df[df["name"] == rec]
        if rec_row.empty:
            continue

        rec_ings = set(
            rec_row.iloc[0]["ingredients_clean"].split()
        )

        overlap_ratio = len(query_ings & rec_ings) / len(query_ings)

        if overlap_ratio >= 0.3:
            match += 1

    return match / k if k > 0 else 0
