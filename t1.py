#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aml_ensemble.data.schemas import NUMERIC_COLS, CATEGORICAL_COLS


# -----------------------------
# Threshold / metrics utilities
# -----------------------------
def threshold_for_alert_rate(y_score: np.ndarray, alert_rate: float) -> float:
    """
    Choose threshold so that approximately top alert_rate fraction are flagged.
    """
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_score)
    if n == 0:
        return 0.5
    k = max(1, int(np.floor(alert_rate * n)))
    thr = np.partition(y_score, -k)[-k]
    return float(thr)


def alert_rate_at_threshold(y_score: np.ndarray, thr: float) -> float:
    y_score = np.asarray(y_score, dtype=float)
    if len(y_score) == 0:
        return float("nan")
    return float((y_score >= thr).mean())


def precision_recall_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score, dtype=float) >= thr).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(precision), float(recall)


def rare_recall_at_threshold(y_true, y_score, thr, rarity_weight):
    """
    Weighted recall over positives only:
      sum(w_i * 1[score_i>=thr]) / sum(w_i)  for y_i==1
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    w = np.asarray(rarity_weight, dtype=float)

    pos = (y_true == 1)
    if pos.sum() == 0:
        return float("nan")

    hit = (y_score >= thr)
    num = float((w[pos] * hit[pos]).sum())
    den = float(w[pos].sum() + 1e-12)
    return num / den


def threshold_for_rare_catcher_min_alert(
    y_true,
    y_score,
    rarity_weight,
    *,
    tol: float = 0.03,
    n_grid: int = 200,
):
    """
    Rare-catcher threshold:
      1) maximize weighted rare recall
      2) among thresholds within (1-tol) of best rare recall, choose minimal alert_rate

    Returns:
      best_thr, best_rare_recall, best_alert_rate
    """
    y_score = np.asarray(y_score, dtype=float)

    if len(y_score) == 0:
        return 0.5, 0.0, 0.0

    # Threshold grid via quantiles (stable & efficient)
    qs = np.linspace(0.0, 1.0, n_grid)
    thr_grid = np.unique(np.quantile(y_score, qs))

    stats = []
    best_rare = -1.0

    for thr in thr_grid:
        rr = rare_recall_at_threshold(y_true, y_score, float(thr), rarity_weight)
        ar = alert_rate_at_threshold(y_score, float(thr))
        if np.isnan(rr) or np.isnan(ar):
            continue
        stats.append((float(thr), float(rr), float(ar)))
        if rr > best_rare:
            best_rare = rr

    if not stats:
        return 0.5, 0.0, 0.0

    target = best_rare * (1.0 - tol)
    feasible = [s for s in stats if s[1] >= target]
    chosen = min(feasible, key=lambda x: x[2])  # minimize alert rate
    return chosen[0], chosen[1], chosen[2]


# -----------------------------
# Data split helpers
# -----------------------------
def parse_year_month_series(s: pd.Series) -> pd.PeriodIndex:
    """
    Accepts YEAR_MONTH formats like:
      - "2024-08"
      - "202408"
      - 202408 (int)
    Returns pandas PeriodIndex with freq='M'
    """
    # Convert to string, strip spaces
    ss = s.astype(str).str.strip()

    # If looks like YYYYMM (6 digits), convert to YYYY-MM
    mask_yyyymm = ss.str.fullmatch(r"\d{6}")
    ss2 = ss.copy()
    ss2.loc[mask_yyyymm] = ss2.loc[mask_yyyymm].str.slice(0, 4) + "-" + ss2.loc[mask_yyyymm].str.slice(4, 6)

    # Now parse to period
    dt = pd.to_datetime(ss2, format="%Y-%m", errors="coerce")
    if dt.isna().any():
        bad = ss[dt.isna()].head(10).tolist()
        raise ValueError(f"Failed to parse YEAR_MONTH for some rows. Examples: {bad}")
    return dt.dt.to_period("M")


def split_train_val_by_last_n_months(df: pd.DataFrame, year_month_col: str, val_months: int):
    ym = parse_year_month_series(df[year_month_col])
    df = df.copy()
    df["_YM_PERIOD_"] = ym

    months_sorted = np.array(sorted(df["_YM_PERIOD_"].unique()))
    if len(months_sorted) < val_months + 1:
        raise ValueError(f"Not enough distinct months to take last {val_months} months as val.")

    val_set = set(months_sorted[-val_months:])
    df_val = df[df["_YM_PERIOD_"].isin(val_set)].copy()
    df_train = df[~df["_YM_PERIOD_"].isin(val_set)].copy()

    # cleanup helper column
    df_train.drop(columns=["_YM_PERIOD_"], inplace=True)
    df_val.drop(columns=["_YM_PERIOD_"], inplace=True)

    return df_train, df_val, months_sorted[-val_months:]


# -----------------------------
# Model loading / scoring
# -----------------------------
def load_selected_models(selected_csv: Path) -> pd.DataFrame:
    df_sel = pd.read_csv(selected_csv)
    required = {"source_target", "model_id", "group"}
    missing = required - set(df_sel.columns)
    if missing:
        raise ValueError(f"selected_csv missing required columns: {sorted(missing)}")
    df_sel["source_target"] = df_sel["source_target"].astype(str).str.lower().str.strip()
    df_sel["group"] = df_sel["group"].astype(str).str.lower().str.strip()
    df_sel["model_id"] = df_sel["model_id"].astype(str).str.strip()
    return df_sel


def load_model_object(source_target: str, model_id: str):
    p = Path("models") / source_target / f"{model_id}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Missing model file: {p}")
    payload = joblib.load(p)
    return payload["model"]


def compute_scores_for_models(X, df_sel: pd.DataFrame) -> np.ndarray:
    """
    Returns scores matrix shape (n_samples, n_models) following df_sel order.
    """
    models = []
    for _, r in df_sel.iterrows():
        models.append(load_model_object(r["source_target"], r["model_id"]))
    scores = np.column_stack([m.predict_score(X) for m in models])
    return scores


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Backfill per-model best thresholds for selected models across BOTH case/ and str/ folders."
    )
    ap.add_argument("--input_csv", required=True, help="Raw data CSV containing YEAR_MONTH + features + label.")
    ap.add_argument(
        "--selected_csv",
        default="debug/str_coverage_both/selected_models_all.csv",
        help="Selected models CSV from analyze_str_coverage_both.py",
    )
    ap.add_argument("--output_csv", default=None, help="Output CSV with thresholds. Default: <selected>_thresholded.csv")

    ap.add_argument("--year_month_col", default="YEAR_MONTH", help="YEAR_MONTH column name.")
    ap.add_argument("--label_col", default="STR_Ind", help="Label to optimize thresholds for (default STR_Ind).")
    ap.add_argument("--val_months", type=int, default=3, help="Use last N months as validation set.")

    # Coverage threshold settings
    ap.add_argument("--alert_rate", type=float, default=0.01, help="Alert budget for coverage models (e.g., 0.01 = 1%).")

    # Rare-catcher threshold settings
    ap.add_argument("--tol", type=float, default=0.03, help="Rare recall tolerance (e.g., 0.03 means within 3% of best).")
    ap.add_argument("--n_grid", type=int, default=200, help="Quantile grid size for threshold scanning.")
    ap.add_argument(
        "--rarity_base_threshold",
        type=float,
        default=0.5,
        help="Base threshold used to estimate rarity_weight via caught_count across selected models.",
    )

    # Debug / performance
    ap.add_argument("--max_val_rows", type=int, default=0, help="Optional cap on val rows for speed (0 = no cap).")
    ap.add_argument("--export_val_summary", action="store_true", help="Export a small summary of val split stats.")
    return ap.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_csv).resolve()
    selected_path = Path(args.selected_csv).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not selected_path.exists():
        raise FileNotFoundError(f"Selected CSV not found: {selected_path}")

    out_path = Path(args.output_csv).resolve() if args.output_csv else selected_path.with_name(selected_path.stem + "_thresholded.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load preprocessor
    preproc_path = Path("metadata") / "preprocessor.pkl"
    if not preproc_path.exists():
        raise FileNotFoundError("metadata/preprocessor.pkl not found. Train initial pool first.")
    pre = joblib.load(preproc_path)

    # Load data
    df = pd.read_csv(input_path)
    if args.year_month_col not in df.columns:
        raise ValueError(f"Missing year_month_col: {args.year_month_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label_col: {args.label_col}")

    feat_cols = NUMERIC_COLS + CATEGORICAL_COLS
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Split
    _, df_val, val_month_list = split_train_val_by_last_n_months(df, args.year_month_col, args.val_months)

    if args.max_val_rows and len(df_val) > args.max_val_rows:
        df_val = df_val.sample(args.max_val_rows, random_state=42)

    y_val = df_val[args.label_col].to_numpy(dtype=int)
    if y_val.sum() == 0:
        raise ValueError(
            f"Validation set has 0 positives for label_col={args.label_col}. "
            f"Try increasing --val_months or choosing different split."
        )

    X_val = pre.transform(df_val[feat_cols])

    # Load selected models (both folders)
    df_sel = load_selected_models(selected_path)

    # Score all selected models on FULL val set (needed for alert_rate calculations)
    scores_val_all = compute_scores_for_models(X_val, df_sel)  # (n_val, n_models)

    # Estimate rarity_weight based on how many selected models catch each positive at a base threshold
    # This makes rarity_weight stable and not dependent on any single model's chosen threshold.
    hits_base = (scores_val_all >= float(args.rarity_base_threshold)).astype(np.int8)
    caught_count = hits_base.sum(axis=1).astype(float)  # per val row
    rarity_weight = 1.0 / (caught_count + 1e-6)

    # (optional) export split summary
    if args.export_val_summary:
        summary = {
            "val_months": args.val_months,
            "val_month_list": [str(m) for m in val_month_list],
            "n_val": int(len(df_val)),
            "pos_in_val": int(y_val.sum()),
            "pos_rate_in_val": float(y_val.mean()),
        }
        pd.DataFrame([summary]).to_csv(out_path.parent / "val_split_summary.csv", index=False)

    # Compute per-model thresholds
    best_thr_list = []
    method_list = []
    alert_rate_list = []
    precision_list = []
    recall_list = []
    rare_recall_list = []

    for j in range(scores_val_all.shape[1]):
        group = df_sel.loc[j, "group"]
        s = scores_val_all[:, j]

        if group != "rare":
            # Coverage models: threshold by alert budget
            thr = threshold_for_alert_rate(s, alert_rate=float(args.alert_rate))
            pr, rc = precision_recall_at_threshold(y_val, s, thr)
            ar = alert_rate_at_threshold(s, thr)

            best_thr_list.append(float(thr))
            method_list.append(f"alert_rate@{args.alert_rate:g}")
            alert_rate_list.append(float(ar))
            precision_list.append(float(pr))
            recall_list.append(float(rc))
            rare_recall_list.append(float("nan"))
        else:
            # Rare-catcher: maximize rare_recall then minimize alert
            thr, rr, ar = threshold_for_rare_catcher_min_alert(
                y_true=y_val,
                y_score=s,
                rarity_weight=rarity_weight,
                tol=float(args.tol),
                n_grid=int(args.n_grid),
            )
            pr, rc = precision_recall_at_threshold(y_val, s, thr)

            best_thr_list.append(float(thr))
            method_list.append(f"rare_min_alert@tol={args.tol:g}")
            alert_rate_list.append(float(ar))
            precision_list.append(float(pr))
            recall_list.append(float(rc))
            rare_recall_list.append(float(rr))

    df_out = df_sel.copy()
    df_out["best_threshold"] = best_thr_list
    df_out["threshold_method"] = method_list
    df_out["val_alert_rate_at_thr"] = alert_rate_list
    df_out["val_precision_at_thr"] = precision_list
    df_out["val_recall_at_thr"] = recall_list
    df_out["val_rare_recall_at_thr"] = rare_recall_list
    df_out["rarity_base_threshold"] = float(args.rarity_base_threshold)
    df_out["val_months_used"] = int(args.val_months)

    df_out.to_csv(out_path, index=False)

    print("[OK] Threshold backfill complete.")
    print(f"  Input data:   {input_path}")
    print(f"  Selected:     {selected_path}")
    print(f"  Output:       {out_path}")
    print(f"  Val months:   {[str(m) for m in val_month_list]}")
    print(f"  Val positives ({args.label_col}): {int(y_val.sum())} / {len(y_val)}")


if __name__ == "__main__":
    main()
