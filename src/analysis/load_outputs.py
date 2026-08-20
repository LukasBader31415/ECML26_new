from pathlib import Path
import pickle
import pandas as pd


def _as_path(path_like) -> Path:
    return Path(path_like)


def _standardize_customer_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "CustomerCode" in df.columns and "customer_id" not in df.columns:
        df = df.rename(columns={"CustomerCode": "customer_id"})
    if "id" in df.columns and "customer_id" not in df.columns:
        df = df.rename(columns={"id": "customer_id"})

    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype(str)

    return df


def load_pointwise_structure(structural_run_dir) -> pd.DataFrame:
    structural_run_dir = _as_path(structural_run_dir)
    path = structural_run_dir / "pointwise_structure.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = _standardize_customer_col(df)

    if "view" not in df.columns:
        raise ValueError("pointwise_structure.csv braucht eine Spalte 'view'.")

    return df


def load_single_view_oof(single_view_run_dir) -> pd.DataFrame:
    single_view_run_dir = _as_path(single_view_run_dir)
    parts = []

    for sub in sorted(single_view_run_dir.iterdir()):
        if not sub.is_dir():
            continue

        path = sub / "oof_predictions.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        df = _standardize_customer_col(df)
        df["view"] = sub.name
        parts.append(df)

    if not parts:
        raise ValueError(f"Keine oof_predictions.csv unter {single_view_run_dir} gefunden.")

    out = pd.concat(parts, ignore_index=True)
    return out


def load_single_view_summary(single_view_run_dir) -> pd.DataFrame:
    single_view_run_dir = _as_path(single_view_run_dir)
    path = single_view_run_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_m3_search_folds(search_runs_dir) -> pd.DataFrame:
    search_runs_dir = _as_path(search_runs_dir)

    aggregated = search_runs_dir / "all_runs_fold_df.csv"
    if aggregated.exists():
        return pd.read_csv(aggregated)

    parts = []
    for sub in sorted(search_runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        path = sub / "fold_df.csv"
        if path.exists():
            parts.append(pd.read_csv(path))

    if not parts:
        raise ValueError(f"Keine fold_df.csv unter {search_runs_dir} gefunden.")

    return pd.concat(parts, ignore_index=True)


def load_m3_fixed_folds(fixed_runs_dir) -> pd.DataFrame:
    fixed_runs_dir = _as_path(fixed_runs_dir)

    aggregated = fixed_runs_dir / "all_fixed_fold_df.csv"
    if aggregated.exists():
        return pd.read_csv(aggregated)

    parts = []
    for sub in sorted(fixed_runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        path = sub / "fold_df.csv"
        if path.exists():
            parts.append(pd.read_csv(path))

    if not parts:
        raise ValueError(f"Keine fold_df.csv unter {fixed_runs_dir} gefunden.")

    return pd.concat(parts, ignore_index=True)


def load_m3_fixed_oof_from_pickles(fixed_runs_dir) -> pd.DataFrame:
    fixed_runs_dir = _as_path(fixed_runs_dir)
    parts = []

    for sub in sorted(fixed_runs_dir.iterdir()):
        if not sub.is_dir():
            continue

        result_path = sub / "result.pkl"
        fold_path = sub / "fold_df.csv"

        if not result_path.exists():
            continue

        with result_path.open("rb") as f:
            res = pickle.load(f)

        if not hasattr(res, "oof_df"):
            continue

        oof_df = res.oof_df.copy()
        oof_df = _standardize_customer_col(oof_df)

        if fold_path.exists():
            fold_df = pd.read_csv(fold_path)
            if len(fold_df) > 0:
                first = fold_df.iloc[0]
                for col in ["profile_name", "K_0", "K_1", "vweight_0", "vweight_1", "vweight_2", "run_name"]:
                    if col in fold_df.columns:
                        oof_df[col] = first[col]

        if "profile_name" not in oof_df.columns:
            oof_df["profile_name"] = sub.name

        parts.append(oof_df)

    if not parts:
        raise ValueError(f"Keine result.pkl mit oof_df unter {fixed_runs_dir} gefunden.")

    out = pd.concat(parts, ignore_index=True)
    return out
