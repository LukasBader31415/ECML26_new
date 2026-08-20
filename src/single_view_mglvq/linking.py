import numpy as np
import pandas as pd


def perf_vs_structure(joined: pd.DataFrame) -> pd.DataFrame:
    j = joined.copy()
    j["difficulty"] = (1 - j["knn_purity"]) + np.maximum(0, -j["mu"])

    out = []

    g = j.groupby("correct")[["knn_purity", "mu", "knn_mean", "d_plus", "d_minus"]].mean()
    for c in [0, 1]:
        if c in g.index:
            out.append({
                "section": "mean_by_correct",
                "group": f"correct={c}",
                **{f"mean_{k}": float(g.loc[c, k]) for k in g.columns},
            })

    def add_err_by_q(col: str, q: int = 5) -> None:
        s = j[col]
        if s.nunique(dropna=True) < 3:
            return
        bins = pd.qcut(s, q=q, duplicates="drop")
        err = j.groupby(bins)["correct"].apply(lambda x: 1 - x.mean())
        for b, v in err.items():
            out.append({
                "section": f"error_by_quantile_{col}",
                "group": str(b),
                "error_rate": float(v),
            })

    add_err_by_q("knn_purity")
    add_err_by_q("mu")
    add_err_by_q("knn_mean")
    add_err_by_q("difficulty")

    return pd.DataFrame(out)


def join_structure_and_predictions(
    pointwise_structure_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    view_name: str,
) -> pd.DataFrame:
    pt = pointwise_structure_df.copy()
    oof = oof_df.copy()

    if "view" in pt.columns:
        pt = pt.loc[pt["view"] == view_name].copy()

    if "customer_id" in pt.columns:
        pt["customer_id"] = pt["customer_id"].astype(str)
        pt = pt.set_index("customer_id")
    elif "id" in pt.columns:
        pt["id"] = pt["id"].astype(str)
        pt = pt.set_index("id")
    else:
        pt.index = pt.index.astype(str)

    if "id" not in oof.columns:
        raise ValueError("oof_df braucht eine Spalte 'id'.")

    oof["id"] = oof["id"].astype(str)

    joined = pt.merge(oof, left_index=True, right_on="id", how="inner")
    return joined