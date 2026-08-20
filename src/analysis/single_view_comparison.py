from itertools import combinations
import numpy as np
import pandas as pd


def infer_views_from_wide(wide_df: pd.DataFrame, prefix: str = "correct_") -> list[str]:
    views = []
    for col in wide_df.columns:
        if col.startswith(prefix):
            views.append(col.replace(prefix, ""))
    return sorted(views)


def _dominance_label(row: pd.Series, views: list[str]) -> str:
    good = [v for v in views if row.get(f"correct_{v}", 0) == 1]
    return "+".join(good) if good else "none"


def build_prediction_dominance_groups(wide_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = wide_df.copy()
    views = infer_views_from_wide(df)

    df["dominance_group"] = df.apply(lambda r: _dominance_label(r, views), axis=1)
    df["n_correct_views"] = df[[f"correct_{v}" for v in views]].fillna(0).sum(axis=1)

    summary = (
        df.groupby("dominance_group", as_index=False)
        .agg(
            n_customers=("customer_id", "size"),
            mean_n_correct_views=("n_correct_views", "mean"),
        )
        .sort_values("n_customers", ascending=False)
        .reset_index(drop=True)
    )
    summary["share"] = summary["n_customers"] / summary["n_customers"].sum()

    return df, summary


def build_structure_prediction_alignment(
    wide_df: pd.DataFrame,
    structure_metric: str = "knn_purity",
) -> pd.DataFrame:
    df = wide_df.copy()
    views = infer_views_from_wide(df)

    metric_cols = [f"{structure_metric}_{v}" for v in views if f"{structure_metric}_{v}" in df.columns]
    if not metric_cols:
        raise ValueError(f"Keine Spalten für structure_metric='{structure_metric}' gefunden.")

    if structure_metric in {"mu", "knn_mean", "d_plus"}:
        best_idx = df[metric_cols].idxmin(axis=1)
    else:
        best_idx = df[metric_cols].idxmax(axis=1)

    df[f"best_structure_view_{structure_metric}"] = best_idx.str.replace(f"{structure_metric}_", "", regex=False)

    df["correct_views_label"] = df.apply(lambda r: _dominance_label(r, views), axis=1)
    df["n_correct_views"] = df[[f"correct_{v}" for v in views]].fillna(0).sum(axis=1)
    df["any_view_correct"] = df["n_correct_views"] > 0

    def _structure_best_correct(row):
        best_view = row[f"best_structure_view_{structure_metric}"]
        return int(row.get(f"correct_{best_view}", 0) == 1)

    df[f"structure_best_correct_{structure_metric}"] = df.apply(_structure_best_correct, axis=1)

    return df


def summarize_view_complementarity(wide_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = wide_df.copy()
    views = infer_views_from_wide(df)

    overall = pd.DataFrame([{
        "n_customers": len(df),
        "share_none_correct": float((df[[f"correct_{v}" for v in views]].fillna(0).sum(axis=1) == 0).mean()),
        "share_exactly_one_correct": float((df[[f"correct_{v}" for v in views]].fillna(0).sum(axis=1) == 1).mean()),
        "share_exactly_two_correct": float((df[[f"correct_{v}" for v in views]].fillna(0).sum(axis=1) == 2).mean()),
        "share_all_correct": float((df[[f"correct_{v}" for v in views]].fillna(0).sum(axis=1) == len(views)).mean()),
    }])

    per_view_rows = []
    for v in views:
        col = f"correct_{v}"
        per_view_rows.append({
            "view": v,
            "accuracy_from_oof": float(df[col].mean()),
            "n_correct": int(df[col].sum()),
        })
    per_view = pd.DataFrame(per_view_rows).sort_values("accuracy_from_oof", ascending=False)

    pair_rows = []
    for a, b in combinations(views, 2):
        ca = df[f"correct_{a}"].fillna(0).astype(int)
        cb = df[f"correct_{b}"].fillna(0).astype(int)

        both_correct = ((ca == 1) & (cb == 1)).sum()
        either_correct = ((ca == 1) | (cb == 1)).sum()
        only_a = ((ca == 1) & (cb == 0)).sum()
        only_b = ((ca == 0) & (cb == 1)).sum()

        pair_rows.append({
            "view_a": a,
            "view_b": b,
            "both_correct": int(both_correct),
            "either_correct": int(either_correct),
            "only_a_correct": int(only_a),
            "only_b_correct": int(only_b),
            "jaccard_correct_sets": float(both_correct / either_correct) if either_correct > 0 else 0.0,
        })

    pairwise = pd.DataFrame(pair_rows).sort_values("jaccard_correct_sets")

    return {
        "overall": overall,
        "per_view": per_view,
        "pairwise": pairwise,
    }
