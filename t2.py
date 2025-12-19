#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aml_ensemble.data.schemas import NUMERIC_COLS, CATEGORICAL_COLS


def parse_args():
    ap = argparse.ArgumentParser(
        description="Score a CSV using selected models from BOTH case/ and str/ with per-model thresholds."
    )
    ap.add_argument("--input_csv", required=True, help="Input CSV with raw features.")
    ap.add_argument(
        "--selected_thresholded_csv",
        default="debug/str_coverage_both/selected_models_all_thresholded.csv",
        help="CSV produced by backfill_thresholds_selected_models_both.py",
    )
    ap.add_argument(
        "--output_csv",
        default=None,
        help="Output CSV. Default: <input>_scored_thresholded_both.csv",
    )

    # Weighting config (same as previous auto-weight baseline)
    ap.add_argument("--rare_multiplier", type=float, default=2.0, help="Multiply weights for group=='rare'.")
    ap.add_argument("--w_val", type=float, default=1.0, help="Weight component coefficient for val_score.")
    ap.add_argument("--w_cov", type=float, default=1.0, help="Weight component coefficient for coverage_pos_rate.")
    ap.add_argument("--w_rare", type=float, default=1.0, help="Weight component coefficient for rare_catch_score.")
    ap.add_argument("--top_n", type=int, default=0, help="Optionally keep only top N models by computed weight (0=all).")

    # Threshold usage mode
    ap.add_argument(
        "--mode",
        choices=["gated", "soft"],
        default="gated",
        help=(
            "gated: model contributes only if score>=its threshold (recommended)\n"
            "soft: ignore thresholds, pure weighted average (legacy)"
        ),
    )
    ap.add_argument(
        "--gate_value",
        choices=["score", "binary"],
        default="score",
        help=(
            "When mode=gated:\n"
            "  score: contribution = weight * score * 1[score>=thr]\n"
            "  binary: contribution = weight * 1[score>=thr]  (pure voting)"
        ),
    )

    # Exports
    ap.add_argument("--export_weight_table", action="store_true", help="Export weights to model_weights_thresholded.csv")
    ap.add_argument(
        "--export_hits",
        action="store_true",
        help="Export per-row which models fired (can be large).",
    )
    ap.add_argument(
        "--hits_max_models",
        type=int,
        default=50,
        help="When export_hits, at most list this many model keys per row.",
    )
    return ap.parse_args()


def _minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(hi - lo, 0.0):
        return np.ones_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _compute_auto_weights(df_sel: pd.DataFrame, w_val: float, w_cov: float, w_rare: float, rare_multiplier: float):
    """
    base = w_val*norm(val_score) + w_cov*norm(coverage_pos_rate) + w_rare*norm(rare_catch_score)
    if group == 'rare': base *= rare_multiplier
    weights = base / sum(base)
    """
    df_sel = df_sel.copy()

    # Defensive defaults
    if "group" not in df_sel.columns:
        df_sel["group"] = "coverage"
    for col in ["val_score", "coverage_pos_rate", "rare_catch_score"]:
        if col not in df_sel.columns:
            df_sel[col] = 0.0

    v = _minmax_normalize(df_sel["val_score"].to_numpy(dtype=float))
    c = _minmax_normalize(df_sel["coverage_pos_rate"].to_numpy(dtype=float))
    r = _minmax_normalize(df_sel["rare_catch_score"].to_numpy(dtype=float))

    base = (w_val * v) + (w_cov * c) + (w_rare * r)
    base = np.where(np.isfinite(base), base, 0.0)

    grp = df_sel["group"].astype(str).str.lower().to_numpy()
    base = np.where(grp == "rare", base * rare_multiplier, base)

    if np.isclose(base.sum(), 0.0):
        base = np.ones_like(base, dtype=float)
    weights = base / base.sum()

    df_sel["norm_val_score"] = v
    df_sel["norm_coverage"] = c
    df_sel["norm_rare_score"] = r
    df_sel["weight_raw"] = base
    df_sel["weight"] = weights
    return df_sel


def _load_selected_thresholded(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"source_target", "model_id", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"selected_thresholded_csv missing required columns: {sorted(missing)}")

    df["source_target"] = df["source_target"].astype(str).str.lower().str.strip()
    df["model_id"] = df["model_id"].astype(str).str.strip()
    df["group"] = df["group"].astype(str).str.lower().str.strip()

    # threshold column: if missing, fallback to 0.5
    if "best_threshold" not in df.columns:
        df["best_threshold"] = 0.5
        df["threshold_method"] = "fallback_0.5"

    # clean
    df["best_threshold"] = pd.to_numeric(df["best_threshold"], errors="coerce").fillna(0.5)

    return df


def _load_models(df_sel: pd.DataFrame):
    loaded = []
    for _, row in df_sel.iterrows():
        src = row["source_target"]
        mid = row["model_id"]
        p = Path("models") / src / f"{mid}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")
        payload = joblib.load(p)
        loaded.append((src, mid, payload["model"]))
    return loaded


def main():
    args = parse_args()

    input_path = Path(args.input_csv).resolve()
    sel_path = Path(args.selected_thresholded_csv).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not sel_path.exists():
        raise FileNotFoundError(f"Selected thresholded CSV not found: {sel_path}")

    out_path = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else input_path.with_name(input_path.stem + "_scored_thresholded_both.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load preprocessor
    preproc_path = Path("metadata") / "preprocessor.pkl"
    if not preproc_path.exists():
        raise FileNotFoundError("metadata/preprocessor.pkl not found. Train initial pool first.")
    pre = joblib.load(preproc_path)

    # Load input
    df = pd.read_csv(input_path)
    feat_cols = NUMERIC_COLS + CATEGORICAL_COLS
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = pre.transform(df[feat_cols])

    # Load selected models with thresholds
    df_sel = _load_selected_thresholded(sel_path)

    # Compute auto weights (you can keep the same weighting recipe as before)
    df_sel = _compute_auto_weights(
        df_sel=df_sel,
        w_val=args.w_val,
        w_cov=args.w_cov,
        w_rare=args.w_rare,
        rare_multiplier=args.rare_multiplier,
    )

    # Optionally keep only top N
    if args.top_n and args.top_n > 0 and len(df_sel) > args.top_n:
        df_sel = df_sel.sort_values("weight", ascending=False).head(args.top_n).reset_index(drop=True)
        w = df_sel["weight"].to_numpy(dtype=float)
        w = w / w.sum() if not np.isclose(w.sum(), 0.0) else np.ones_like(w) / len(w)
        df_sel["weight"] = w

    # Export weight table
    if args.export_weight_table:
        (out_path.parent / "model_weights_thresholded.csv").write_text(
            df_sel.sort_values("weight", ascending=False).to_csv(index=False),
            encoding="utf-8",
        )

    # Load models (case+str mixed)
    loaded = _load_models(df_sel)

    weights = df_sel["weight"].to_numpy(dtype=float)
    thresholds = df_sel["best_threshold"].to_numpy(dtype=float)
    groups = df_sel["group"].astype(str).str.lower().to_numpy()
    sources = df_sel["source_target"].astype(str).str.lower().to_numpy()

    model_keys = [f"{sources[i]}:{df_sel.loc[i, 'model_id']}" for i in range(len(df_sel))]

    # Score: (n_samples, n_models)
    scores = np.column_stack([m.predict_score(X) for (_, _, m) in loaded])

    # Hits per-model per-row
    hits = (scores >= thresholds[None, :]).astype(np.int8)

    # --- Final score ---
    if args.mode == "soft":
        # Legacy: weighted average of scores
        final_score = (scores * weights[None, :]).sum(axis=1)

    else:
        # Gated: only contributing models count
        if args.gate_value == "binary":
            contrib = hits.astype(float) * weights[None, :]
        else:
            # default: score * hit
            contrib = (scores * hits.astype(float)) * weights[None, :]

        # In gated mode, you can either:
        #  (A) keep weights absolute (missing votes reduce total) -> "conservative"
        #  (B) renormalize by active weights -> "conditional average"
        #
        # I recommend (B): stable scale across rows.
        active_w = (weights[None, :] * hits).sum(axis=1)  # sum of weights that fired
        final_score = contrib.sum(axis=1) / (active_w + 1e-12)

    # Group scores (coverage vs rare), also gated/soft consistent
    cov_mask = (groups != "rare")
    rare_mask = (groups == "rare")

    def group_score(mask: np.ndarray):
        if not mask.any():
            return np.full((scores.shape[0],), np.nan)

        s = scores[:, mask]
        h = hits[:, mask]
        w = weights[mask]

        if args.mode == "soft":
            w2 = w / (w.sum() + 1e-12)
            return (s * w2[None, :]).sum(axis=1)

        # gated
        if args.gate_value == "binary":
            contrib_g = h.astype(float) * w[None, :]
        else:
            contrib_g = (s * h.astype(float)) * w[None, :]

        active = (w[None, :] * h).sum(axis=1)
        return contrib_g.sum(axis=1) / (active + 1e-12)

    coverage_score = group_score(cov_mask)
    rare_score = group_score(rare_mask)

    # Source-level scores: case-only, str-only (optional but useful)
    case_mask = (sources == "case")
    str_mask = (sources == "str")
    case_score = group_score(case_mask)
    str_score = group_score(str_mask)

    # Diagnostics: how many models fired
    total_hit_count = hits.sum(axis=1)
    rare_hit_count = hits[:, rare_mask].sum(axis=1) if rare_mask.any() else np.zeros_like(total_hit_count)
    cov_hit_count = hits[:, cov_mask].sum(axis=1) if cov_mask.any() else np.zeros_like(total_hit_count)

    # Write output
    df_out = df.copy()
    df_out["FINAL_SCORE_ALL"] = final_score
    df_out["COVERAGE_SCORE_ALL"] = coverage_score
    df_out["RARE_SCORE_ALL"] = rare_score
    df_out["CASE_SCORE_ALL"] = case_score
    df_out["STR_SCORE_ALL"] = str_score

    df_out["HIT_COUNT_ALL"] = total_hit_count
    df_out["HIT_COUNT_COVERAGE"] = cov_hit_count
    df_out["HIT_COUNT_RARE"] = rare_hit_count

    df_out.to_csv(out_path, index=False)

    # Optional: per-row list of which models fired
    if args.export_hits:
        # build a compact list per row
        # NOTE: can be expensive for very large datasets
        fired_lists = []
        max_m = int(args.hits_max_models)

        for i in range(hits.shape[0]):
            idxs = np.where(hits[i] == 1)[0]
            if idxs.size > max_m:
                idxs = idxs[:max_m]
            fired_lists.append("|".join([model_keys[j] for j in idxs]))

        hits_path = out_path.parent / "row_fired_models.csv"
        id_cols = [c for c in ["CLNT_NO", "YEAR_MONTH"] if c in df.columns]
        base = df[id_cols].copy() if id_cols else pd.DataFrame({"row_idx": np.arange(len(df))})
        base["hit_count_all"] = total_hit_count
        base["fired_models"] = fired_lists
        base.to_csv(hits_path, index=False)

    print("[OK] Scoring complete with per-model thresholds.")
    print(f"  Input:   {input_path}")
    print(f"  Selected thresholded: {sel_path}")
    print(f"  Output:  {out_path}")
    if args.export_hits:
        print(f"  Fired-model list: {out_path.parent / 'row_fired_models.csv'}")
    if args.export_weight_table:
        print(f"  Weight table: {out_path.parent / 'model_weights_thresholded.csv'}")
    print(f"  Mode: {args.mode} (gate_value={args.gate_value})")
    print("  Added columns: FINAL_SCORE_ALL, COVERAGE_SCORE_ALL, RARE_SCORE_ALL, CASE_SCORE_ALL, STR_SCORE_ALL, HIT_COUNT_*")


if __name__ == "__main__":
    main()
