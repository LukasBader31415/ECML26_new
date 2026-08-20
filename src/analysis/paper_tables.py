import numpy as np
import pandas as pd
from pathlib import Path


def _offdiag_values(D) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Matrix muss quadratisch sein, got {D.shape}")

    mask = np.triu(np.ones_like(D, dtype=bool), k=1)
    vals = D[mask]
    vals = vals[np.isfinite(vals)]
    return vals


def build_distance_distribution_table(
    D_views_norm: dict[str, np.ndarray],
    label_map: dict[str, str] | None = None,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """
    Erzeugt eine Paper-Tabelle der Off-Diagonal-Distanzverteilung
    für normalisierte Matrizen.

    Erwartet z.B.:
        {
            "NAICS": D_naics_normalized,
            "HS": D_hs_normalized,
            "AM": D_am_normalized,
        }
    """
    if label_map is None:
        label_map = {
            "NAICS": "Industry (NAICS)",
            "HS": "Products (HS)",
            "AM": "Applications (AP)",
        }

    rows = []

    for key, D in D_views_norm.items():
        vals = _offdiag_values(D)

        rows.append({
            "Matrix": label_map.get(key, key),
            "Mean": float(vals.mean()),
            "Std.": float(vals.std()),
            "Median": float(np.median(vals)),
            "q0.1": float(np.quantile(vals, 0.10)),
            "q0.9": float(np.quantile(vals, 0.90)),
            "Share(d = 1)": float(np.mean(np.isclose(vals, 1.0, atol=atol))),
        })

    out = pd.DataFrame(rows)

    desired_order = [
        label_map.get("NAICS", "NAICS"),
        label_map.get("HS", "HS"),
        label_map.get("AM", "AM"),
    ]
    out["__order"] = out["Matrix"].map({k: i for i, k in enumerate(desired_order)})
    out = out.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    return out



def _default_view_label_map():
    return {
        "naics": "Industry (NAICS)",
        "hs": "Products (HS)",
        "am": "Applications (AP)",
    }


def build_pearson_correlation_table(
    structural_run_dir,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    structural_run_dir = Path(structural_run_dir)
    path = structural_run_dir / "cross_view" / "correlation_pearson.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    if label_map is None:
        label_map = _default_view_label_map()

    df = pd.read_csv(path, index_col=0)
    df.index = [label_map.get(x, x) for x in df.index]
    df.columns = [label_map.get(x, x) for x in df.columns]

    desired_order = [
        label_map.get("naics", "naics"),
        label_map.get("hs", "hs"),
        label_map.get("am", "am"),
    ]
    df = df.loc[desired_order, desired_order]
    return df


def build_knn_overlap_table(
    structural_run_dir,
    k: int = 10,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    structural_run_dir = Path(structural_run_dir)
    path = structural_run_dir / "cross_view" / f"knn_overlap_k{k}.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    if label_map is None:
        label_map = _default_view_label_map()

    df = pd.read_csv(path, index_col=0)
    df.index = [label_map.get(x, x) for x in df.index]
    df.columns = [label_map.get(x, x) for x in df.columns]

    desired_order = [
        label_map.get("naics", "naics"),
        label_map.get("hs", "hs"),
        label_map.get("am", "am"),
    ]
    df = df.loc[desired_order, desired_order]
    return df


def build_agreement_table(
    structural_run_dir,
    quantile: float = 0.10,
    subset: str = "all",
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    structural_run_dir = Path(structural_run_dir)
    path = structural_run_dir / "cross_view" / f"agreement_q{quantile:.2f}.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    if label_map is None:
        label_map = _default_view_label_map()

    df = pd.read_csv(path)

    # robust gegen csv mit oder ohne explizite Indexspalten
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    needed = {"A", "B", "subset", "agree_similar", "agree_dissimilar", "A_only_similar", "B_only_similar"}
    if not needed.issubset(df.columns):
        raise ValueError(f"agreement file hat nicht die erwarteten Spalten. Vorhanden: {list(df.columns)}")

    df = df[df["subset"] == subset].copy()

    df["A_label"] = df["A"].map(label_map).fillna(df["A"])
    df["B_label"] = df["B"].map(label_map).fillna(df["B"])
    df["Pair (A-B)"] = df["A_label"] + "–" + df["B_label"]

    out = df[[
        "Pair (A-B)",
        "agree_similar",
        "agree_dissimilar",
        "A_only_similar",
        "B_only_similar",
    ]].rename(columns={
        "agree_similar": "Agree Similar",
        "agree_dissimilar": "Agree Dissimilar",
        "A_only_similar": "A-only Similar",
        "B_only_similar": "B-only Similar",
    })

    # gleiche Reihenfolge wie im Paper-Beispiel
    order = {
        "Products (HS)–Applications (AP)": 0,
        "Industry (NAICS)–Applications (AP)": 1,
        "Industry (NAICS)–Products (HS)": 2,
    }
    out["__order"] = out["Pair (A-B)"].map(order).fillna(999)
    out = out.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    return out




def build_structural_characteristics_table(
    single_summary_df: pd.DataFrame,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Erzeugt die Paper-Tabelle:
    Matrix | Sep. Ratio | kNN Pur. | μ | Neg. μ | Intr. Dim.
    """
    if label_map is None:
        label_map = {
            "naics": "Industry (NAICS)",
            "hs": "Products (HS)",
            "am": "Applications (AP)",
        }

    needed = {
        "matrix",
        "sep_ratio_inter_intra",
        "knn_purity_mean",
        "margin_mu_mean",
        "margin_frac_neg",
        "intrinsic_dim_2nn",
    }
    missing = needed - set(single_summary_df.columns)
    if missing:
        raise ValueError(f"single_summary_df fehlt: {missing}")

    out = pd.DataFrame({
        "Matrix": single_summary_df["matrix"].map(label_map).fillna(single_summary_df["matrix"]),
        "Sep. Ratio": single_summary_df["sep_ratio_inter_intra"],
        "kNN Pur.": single_summary_df["knn_purity_mean"],
        "μ": single_summary_df["margin_mu_mean"],
        "Neg. μ": single_summary_df["margin_frac_neg"],
        "Intr. Dim.": single_summary_df["intrinsic_dim_2nn"],
    })

    desired_order = [
        label_map.get("naics", "naics"),
        label_map.get("hs", "hs"),
        label_map.get("am", "am"),
    ]
    out["__order"] = out["Matrix"].map({k: i for i, k in enumerate(desired_order)})
    out = out.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    return out




def build_single_view_baseline_table(
    single_view_summary_df: pd.DataFrame,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Table 6:
    Matrix | K | Bal. Acc. | Recall | TP | FP | TN | FN
    """
    if label_map is None:
        label_map = {
            "naics": "Industry (NAICS)",
            "hs": "Products (HS)",
            "am": "Applications (AP)",
        }

    needed = {
        "matrix",
        "best_K",
        "balanced_accuracy_from_oof",
        "recall_from_oof",
        "tp",
        "fp",
        "tn",
        "fn",
    }
    missing = needed - set(single_view_summary_df.columns)
    if missing:
        raise ValueError(f"single_view_summary_df fehlt: {missing}")

    out = pd.DataFrame({
        "Matrix": single_view_summary_df["matrix"].map(label_map).fillna(single_view_summary_df["matrix"]),
        "K": single_view_summary_df["best_K"].astype(int),
        "Bal. Acc.": single_view_summary_df["balanced_accuracy_from_oof"],
        "Recall": single_view_summary_df["recall_from_oof"],
        "TP": single_view_summary_df["tp"].astype(int),
        "FP": single_view_summary_df["fp"].astype(int),
        "TN": single_view_summary_df["tn"].astype(int),
        "FN": single_view_summary_df["fn"].astype(int),
    })

    desired_order = [
        label_map.get("naics", "naics"),
        label_map.get("hs", "hs"),
        label_map.get("am", "am"),
    ]
    out["__order"] = out["Matrix"].map({k: i for i, k in enumerate(desired_order)})
    out = out.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    return out


def build_structure_prediction_dominance_table(
    wide_df: pd.DataFrame,
    structure_metric: str = "knn_purity",
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Table 7:
    rows    = prediction dominance
    columns = structural dominance

    Prediction dominance:
      - unique correct view -> Industry / Products / Applications
      - exactly two correct -> Mixed
      - all three correct   -> All
      - none correct        -> None

    Structural dominance:
      - unique highest local structure metric -> Industry / Products / Applications
      - ties for highest                      -> Mixed
    """
    if label_map is None:
        label_map = {
            "naics": "Industry (NAICS)",
            "hs": "Products (HS)",
            "am": "Applications (AP)",
        }

    df = wide_df.copy()

    # infer views from correct_* columns
    views = sorted([c.replace("correct_", "") for c in df.columns if c.startswith("correct_")])
    if not views:
        raise ValueError("wide_df enthält keine correct_* Spalten.")

    # ---------- prediction dominance ----------
    def prediction_group(row):
        correct_views = [v for v in views if row.get(f"correct_{v}", 0) == 1]
        n = len(correct_views)

        if n == 0:
            return "None"
        if n == len(views):
            return "All"
        if n == 1:
            return label_map.get(correct_views[0], correct_views[0])
        return "Mixed"

    df["prediction_group"] = df.apply(prediction_group, axis=1)

    # ---------- structural dominance (tie-aware) ----------
    metric_cols = [f"{structure_metric}_{v}" for v in views]
    missing_metric = [c for c in metric_cols if c not in df.columns]
    if missing_metric:
        raise ValueError(f"wide_df fehlt Strukturspalten: {missing_metric}")

    def structure_group(row):
        vals = {v: row[f"{structure_metric}_{v}"] for v in views}
        max_val = max(vals.values())
        winners = [v for v, val in vals.items() if pd.notna(val) and np.isclose(val, max_val)]

        if len(winners) == 1:
            return label_map.get(winners[0], winners[0])
        return "Mixed"

    df["structure_group"] = df.apply(structure_group, axis=1)

    row_order = [
        label_map.get("naics", "naics"),
        label_map.get("hs", "hs"),
        label_map.get("am", "am"),
        "Mixed",
        "All",
        "None",
    ]
    col_order = [
        label_map.get("naics", "naics"),
        label_map.get("hs", "hs"),
        label_map.get("am", "am"),
        "Mixed",
    ]

    table = pd.crosstab(
        df["prediction_group"],
        df["structure_group"],
        dropna=False,
    )

    table = table.reindex(index=row_order, columns=col_order, fill_value=0)
    table.index.name = "Prediction Best \\ Structure Best"

    return table.reset_index()


def build_fixed_profile_dominance_table(
    wide_df: pd.DataFrame,
    fixed_oof_df: pd.DataFrame,
    profile_map: dict[str, str],
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Table 9:
    rows = single-view prediction-dominance groups
    columns = selected fixed-weight M3GLVQ variants

    profile_map:
        {
            "Cluster 0": "C0_primary",
            "Cluster 3": "C3_primary",
            "Cluster 4": "C4_primary",
        }
    """
    if label_map is None:
        label_map = {
            "naics": "Industry (NAICS)",
            "hs": "Products (HS)",
            "am": "Applications (AP)",
        }

    wide = wide_df.copy()
    fixed = fixed_oof_df.copy()

    wide["customer_id"] = wide["customer_id"].astype(str)
    fixed["customer_id"] = fixed["customer_id"].astype(str)

    if "profile_name" not in fixed.columns:
        raise ValueError("fixed_oof_df braucht eine Spalte 'profile_name'.")
    if "correct" not in fixed.columns:
        raise ValueError("fixed_oof_df braucht eine Spalte 'correct'.")

    views = sorted([c.replace("correct_", "") for c in wide.columns if c.startswith("correct_")])
    if not views:
        raise ValueError("wide_df enthält keine correct_* Spalten.")

    def dominance_group(row):
        correct_views = [v for v in views if row.get(f"correct_{v}", 0) == 1]
        n = len(correct_views)

        if n == 0:
            return "All wrong"
        if n == len(views):
            return "All correct"
        if n == 1:
            return label_map.get(correct_views[0], correct_views[0])
        return "Mixed"

    wide["dominance_group_table9"] = wide.apply(dominance_group, axis=1)

    row_order = [
        label_map.get("naics", "naics"),
        label_map.get("hs", "hs"),
        label_map.get("am", "am"),
        "Mixed",
        "All correct",
        "All wrong",
    ]

    group_sizes = (
        wide.groupby("dominance_group_table9", as_index=False)
        .agg(**{"Nr. of Customers": ("customer_id", "size")})
    )
    group_sizes = group_sizes.set_index("dominance_group_table9").reindex(row_order).fillna(0)
    group_sizes["Nr. of Customers"] = group_sizes["Nr. of Customers"].astype(int)

    out = pd.DataFrame({
        "Single-view prediction-dominance group": row_order,
        "Nr. of Customers": [int(group_sizes.loc[g, "Nr. of Customers"]) for g in row_order],
    })

    baseline_map = {
        g: (0 if g == "All wrong" else int(group_sizes.loc[g, "Nr. of Customers"]))
        for g in row_order
    }

    for display_name, profile_name in profile_map.items():
        g = fixed[fixed["profile_name"] == profile_name].copy()

        merged = g.merge(
            wide[["customer_id", "dominance_group_table9"]],
            on="customer_id",
            how="inner",
        )

        counts = (
            merged.groupby("dominance_group_table9", as_index=True)["correct"]
            .sum()
            .reindex(row_order)
            .fillna(0)
            .astype(int)
        )

        col_vals = []
        for grp in row_order:
            count = int(counts.loc[grp])
            delta = count - baseline_map[grp]
            col_vals.append(f"{count} ({delta:+d})")

        out[display_name] = col_vals

    total_row = {
        "Single-view prediction-dominance group": "Total",
        "Nr. of Customers": int(len(wide)),
    }

    for display_name, profile_name in profile_map.items():
        g = fixed[fixed["profile_name"] == profile_name].copy()
        total_row[display_name] = int(g["correct"].sum())

    out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)
    return out