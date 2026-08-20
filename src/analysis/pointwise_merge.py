import pandas as pd


def merge_structure_and_single_view(
    structure_df: pd.DataFrame,
    oof_df: pd.DataFrame,
) -> pd.DataFrame:
    s = structure_df.copy()
    o = oof_df.copy()

    s["customer_id"] = s["customer_id"].astype(str)
    o["customer_id"] = o["customer_id"].astype(str)

    needed_s = {"customer_id", "view"}
    needed_o = {"customer_id", "view", "y_true", "y_pred", "correct"}

    missing_s = needed_s - set(s.columns)
    missing_o = needed_o - set(o.columns)

    if missing_s:
        raise ValueError(f"structure_df fehlt: {missing_s}")
    if missing_o:
        raise ValueError(f"oof_df fehlt: {missing_o}")

    joined = s.merge(
        o,
        on=["customer_id", "view"],
        how="inner",
        suffixes=("", "_pred"),
    )

    return joined


def build_single_view_wide(joined_df: pd.DataFrame) -> pd.DataFrame:
    df = joined_df.copy()

    views = sorted(df["view"].unique().tolist())

    value_cols = [
        c for c in [
            "y",
            "mu",
            "d_plus",
            "d_minus",
            "knn_purity",
            "knn_mean",
            "fold",
            "y_true",
            "y_pred",
            "correct",
        ]
        if c in df.columns
    ]

    wide = (
        df.set_index(["customer_id", "view"])[value_cols]
        .unstack("view")
    )

    wide.columns = [f"{col}_{view}" for col, view in wide.columns]
    wide = wide.reset_index()

    y_cols = [f"y_{v}" for v in views if f"y_{v}" in wide.columns]
    y_true_cols = [f"y_true_{v}" for v in views if f"y_true_{v}" in wide.columns]

    if y_cols:
        wide["y_ref"] = wide[y_cols].bfill(axis=1).iloc[:, 0]
    if y_true_cols:
        wide["y_true_ref"] = wide[y_true_cols].bfill(axis=1).iloc[:, 0]

    return wide
