import pandas as pd


def standardize_single_view_summary(single_view_summary_df: pd.DataFrame) -> pd.DataFrame:
    df = single_view_summary_df.copy()

    out = pd.DataFrame({
        "model_family": "single_view",
        "model_name": df["matrix"],
        "view_or_cluster": df["matrix"],
        "balanced_accuracy": df["balanced_accuracy_from_oof"],
        "recall": df["recall_from_oof"],
        "precision": df["precision_from_oof"],
        "f1": df["f1_from_oof"],
    })

    return out


def standardize_fixed_profile_summary(summary_fixed_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_fixed_df.copy()

    out = pd.DataFrame({
        "model_family": "multi_view_fixed_profile",
        "model_name": df["profile_name"],
        "view_or_cluster": df["profile_name"],
        "balanced_accuracy": df["balanced_accuracy_mean"],
        "recall": df["recall_mean"],
        "precision": df["precision_mean"],
        "f1": df["f1_mean"],
        "K_0": df["K_0"],
        "K_1": df["K_1"],
    })

    return out


def standardize_cluster_core_df(cluster_core_df: pd.DataFrame) -> pd.DataFrame:
    df = cluster_core_df.copy()

    out = pd.DataFrame({
        "model_family": "multi_view_cluster",
        "model_name": "cluster_" + df["cluster"].astype(str),
        "view_or_cluster": "cluster_" + df["cluster"].astype(str),
        "balanced_accuracy": df["mean_balanced_accuracy_core"],
        "recall": df["mean_recall_core"],
        "precision": pd.NA,
        "f1": pd.NA,
        "K_0": df["K_0_mean"],
        "K_1": df["K_1_mean"],
        "vweight_0": df["vweight_0_mean"],
        "vweight_1": df["vweight_1_mean"],
        "vweight_2": df["vweight_2_mean"],
    })

    return out


def build_paper_performance_comparison(
    single_view_summary_df: pd.DataFrame,
    summary_fixed_df: pd.DataFrame | None = None,
    cluster_core_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    blocks = [standardize_single_view_summary(single_view_summary_df)]

    if summary_fixed_df is not None and len(summary_fixed_df) > 0:
        blocks.append(standardize_fixed_profile_summary(summary_fixed_df))

    if cluster_core_df is not None and len(cluster_core_df) > 0:
        blocks.append(standardize_cluster_core_df(cluster_core_df))

    out = pd.concat(blocks, ignore_index=True)

    best_single_bal_acc = out.loc[out["model_family"] == "single_view", "balanced_accuracy"].max()
    best_single_recall = out.loc[out["model_family"] == "single_view", "recall"].max()

    out["delta_vs_best_single_bal_acc"] = out["balanced_accuracy"] - best_single_bal_acc
    out["delta_vs_best_single_recall"] = out["recall"] - best_single_recall

    out = out.sort_values(
        ["balanced_accuracy", "recall"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return out


def build_fixed_profile_gain_loss(
    fixed_oof_df: pd.DataFrame,
    single_view_wide_df: pd.DataFrame,
) -> pd.DataFrame:
    wide = single_view_wide_df.copy()
    fixed = fixed_oof_df.copy()

    wide["customer_id"] = wide["customer_id"].astype(str)
    fixed["customer_id"] = fixed["customer_id"].astype(str)

    view_cols = sorted([c.replace("correct_", "") for c in wide.columns if c.startswith("correct_")])

    if "correct" not in fixed.columns:
        raise ValueError("fixed_oof_df braucht eine Spalte 'correct'.")
    if "profile_name" not in fixed.columns:
        raise ValueError("fixed_oof_df braucht eine Spalte 'profile_name'.")

    rows = []

    best_single_view = None
    best_single_score = -1
    for v in view_cols:
        s = wide[f"correct_{v}"].mean()
        if s > best_single_score:
            best_single_score = s
            best_single_view = v

    wide["any_single_correct"] = wide[[f"correct_{v}" for v in view_cols]].fillna(0).max(axis=1)

    for profile_name, g in fixed.groupby("profile_name"):
        merged = g.merge(wide, on="customer_id", how="inner", suffixes=("_fixed", ""))

        row = {
            "profile_name": profile_name,
            "n_samples": len(merged),
        }

        row["gain_vs_any_single"] = int(((merged["correct"] == 1) & (merged["any_single_correct"] == 0)).sum())
        row["loss_vs_any_single"] = int(((merged["correct"] == 0) & (merged["any_single_correct"] == 1)).sum())
        row["net_vs_any_single"] = row["gain_vs_any_single"] - row["loss_vs_any_single"]

        if best_single_view is not None:
            ref_col = f"correct_{best_single_view}"
            row["best_single_view_ref"] = best_single_view
            row["gain_vs_best_single_view"] = int(((merged["correct"] == 1) & (merged[ref_col] == 0)).sum())
            row["loss_vs_best_single_view"] = int(((merged["correct"] == 0) & (merged[ref_col] == 1)).sum())
            row["net_vs_best_single_view"] = row["gain_vs_best_single_view"] - row["loss_vs_best_single_view"]

        for v in view_cols:
            ref_col = f"correct_{v}"
            row[f"gain_vs_{v}"] = int(((merged["correct"] == 1) & (merged[ref_col] == 0)).sum())
            row[f"loss_vs_{v}"] = int(((merged["correct"] == 0) & (merged[ref_col] == 1)).sum())
            row[f"net_vs_{v}"] = row[f"gain_vs_{v}"] - row[f"loss_vs_{v}"]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("net_vs_any_single", ascending=False).reset_index(drop=True)
