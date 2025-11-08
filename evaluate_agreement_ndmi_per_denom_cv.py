#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_agreement_ndmi_per_denom_cv.py
=======================================
Evaluate agreement with expert 3-class labels using **one denominator at a time**
as a single 1D feature (per-case mean NDMI), with repeated Stratified K-Fold and
strict out-of-fold (OOF) evaluation to avoid leakage.

Inputs (produced by evaluate_MI_jacobian_NDMI_multi_denoms.py):
  ./NDMI_multiDen_<FLOW_METHOD>/<noise>/<case>_NDMI-<denom>.npy
Where <denom> in {orig, clip, pct, mad}. Each file is a 1D array over frames.

For each denom separately, this script:
  - Builds a dataset with X = per-case mean NDMI (shape n×1), y = expert label in {1,2,3}.
  - Runs repeated Stratified K-Fold CV with classifier (rf/svm/xgb).
  - Records OOF predictions/probabilities and per-fold metrics.
  - Aggregates OOF (majority vote) to compute Accuracy, quadratic-weighted Kappa, Spearman,
    with 95% bootstrap CIs and permutation p-values.
  - Saves confusion matrices and one-vs-rest PR curves (using OOF probabilities).

Outputs under: ./agreement_per_denom_cv_<FLOW_METHOD>/denom-<DENOM>/
  - oof_predictions.csv
  - fold_metrics.csv
  - aggregate_metrics.csv
  - confusion_counts_oof.png
  - confusion_rownorm_oof.png
  - pr_curves_oof.png
Also writes a summary across denoms: ./agreement_per_denom_cv_<FLOW_METHOD>/aggregate_metrics_all_denoms.csv

Usage (defaults are sensible):
  python evaluate_agreement_ndmi_per_denom_cv.py \
    --flow_method flow_INR \
    --classifier svm \
    --n_splits 5 --n_repeats 50 --seed 42
"""

import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any, List
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, auc, cohen_kappa_score,
                             accuracy_score)
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# xgboost is optional; guard import for environments without it
try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    XGBClassifier = None  # type: ignore
    _HAS_XGB = False

# ----------------------- Expert labels ----------------
EXPERT = {
    "1-1": 2,
    "2-1": 3,
    "3-1": 3,
    "4-1": 2,
    "5-1": 1,
    "6-1": 2,
    "7-1": 3,
    "8-1": 1,
    "9-1": 3,
    "10-1": 1,
    "11-1": 2,
    "12-1": 1,
    "1-2": 3,
    "2-2": 3,
    "3-2": 2,
    "6-2": 1,
    "9-2": 3,
    "11-2": 2,
    "9-3": 3,
}
CASE_RE = re.compile(r"(\d+-\d+)")
DENOMS = ["orig", "clip", "pct", "mad"]

# ----------------------- Helpers ----------------------
def extract_base_case(case_name: str, strict: bool) -> Optional[str]:
    """Return exact match if present; otherwise extract first \\d+-\\d+ substring."""
    if case_name in EXPERT:
        return case_name
    if strict:
        return None
    m = CASE_RE.search(case_name)
    return m.group(1) if m else None


def load_dataset_means(in_root: Path, strict: bool) -> pd.DataFrame:
    """Load per-case mean NDMI for each denom and expert labels into a DataFrame."""
    data = []
    for noise_dir in sorted([d for d in in_root.iterdir() if d.is_dir()]):
        noise = noise_dir.name
        for f in sorted(noise_dir.glob("*_NDMI-orig.npy")):
            case_name = f.name.replace("_NDMI-orig.npy", "")
            base = extract_base_case(case_name, strict)
            if base is None or base not in EXPERT:
                continue
            row = {"noise": noise, "case": case_name, "base_case": base}
            for d in DENOMS:
                p = in_root / noise / f"{case_name}_NDMI-{d}.npy"
                if p.exists():
                    arr = np.load(p)
                    row[d] = float(np.mean(arr)) if arr.size > 0 else np.nan
                else:
                    row[d] = np.nan
            data.append(row)
    df = pd.DataFrame(data)
    df["expert"] = df["base_case"].map(EXPERT).astype(int)
    # Keep rows where at least one denom is present
    keep_mask = df[DENOMS].notna().any(axis=1)
    df = df[keep_mask].reset_index(drop=True)
    return df


def get_classifier(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=None,
                                      min_samples_leaf=1, random_state=seed, class_weight="balanced")
    if kind == "svm":
        return SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=seed, class_weight="balanced")
    if kind == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("xgboost is not installed; choose --classifier rf or svm")
        return XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
            eval_metric="mlogloss", random_state=seed, num_class=3
        )
    raise ValueError(f"Unknown classifier: {kind}")


def _majority_vote_safe(labels_int_array: np.ndarray) -> int:
    """Return majority vote among {1,2,3}; ties resolved by smallest class id; never returns 0."""
    counts = np.bincount(labels_int_array, minlength=4)  # index 0 unused
    idx = np.argmax(counts[1:]) + 1
    return int(idx)


def bootstrap_ci(metric_func, y_true, y_pred, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_func(y_true[idx], y_pred[idx]))
    stats = np.array(stats, dtype=float)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def metric_spearman(y_true, y_pred):
    r, _ = spearmanr(y_true, y_pred)
    return float(0.0 if np.isnan(r) else r)


def perm_pvalue(metric_func, y_true, y_pred, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = y_true.copy()
    obs = metric_func(y_true, y_pred)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(y_true)
        val = metric_func(y_true, y_pred)
        if val >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def plot_confusions(y_true, y_pred, outdir: Path, tag: str = "oof"):
    labels = [1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Confusion (counts) — {tag}")
    fig.tight_layout()
    fig.savefig(outdir / f"confusion_counts_{tag}.png", dpi=160)
    plt.close(fig)
    cm_row = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm_row, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm_row.shape[0]):
        for j in range(cm_row.shape[1]):
            ax.text(j, i, f"{cm_row[i, j] * 100:.1f}%", ha="center", va="center")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Expert")
    ax.set_title(f"Confusion (row-normalized) — {tag}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / f"confusion_rownorm_{tag}.png", dpi=160)
    plt.close(fig)


def plot_pr_oof(y_true, proba, outdir: Path, tag: str = "oof"):
    classes = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    auprs = {}
    for k, cls in enumerate(classes, start=0):
        y_pos = (y_true == cls).astype(int)
        prec, rec, _ = precision_recall_curve(y_pos, proba[:, k])
        aupr = auc(rec, prec)
        auprs[cls] = float(aupr)
        ax.plot(rec, prec, label=f"class {cls} (AUPRC={aupr:.3f})", linewidth=2.0)
        # ax.step(rec, prec, where="post", label=f"class {cls} (AUPRC={aupr:.3f})", linewidth=2.0)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"One-vs-rest PR — {tag}")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / f"pr_curves_{tag}.png", dpi=160)
    plt.close(fig)
    return auprs


def run_one_denom(df: pd.DataFrame, denom: str, out_root: Path,
                  classifier: str, n_splits: int, n_repeats: int, seed: int) -> Dict[str, Any]:
    """Run CV and OOF evaluation for a single denom; save files to denom-specific folder."""
    # Keep only rows with this denom present
    dfx = df.dropna(subset=[denom]).reset_index(drop=True)
    X = dfx[[denom]].to_numpy()   # (n,1)  single feature
    y = dfx["expert"].to_numpy().astype(int)
    cases = dfx["case"].tolist()

    outdir = out_root / f"denom-{denom}"
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    all_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=int(rng.integers(0, 1_000_000)))
        fold_id = 0
        for tr_idx, va_idx in skf.split(X, y):
            fold_id += 1
            clf = get_classifier(classifier, int(rng.integers(0, 1_000_000)))
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            clf.fit(Xtr, ytr)
            yhat = clf.predict(Xva)
            # ensure probabilities for classes [1,2,3]
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(Xva)
                classes = getattr(clf, "classes_", np.array([1,2,3]))
                proba_full = np.zeros((len(yva), 3), dtype=float)
                for j, cls in enumerate(classes):
                    if cls in [1,2,3]:
                        proba_full[:, int(cls)-1] = proba[:, j]
                proba = proba_full
            else:
                proba = np.full((len(yva), 3), 1.0/3.0, dtype=float)

            # store rows
            for i, idx in enumerate(va_idx):
                all_rows.append({
                    "repeat": rep, "fold": fold_id, "index": int(idx),
                    "case": cases[idx],
                    "expert": int(yva[i]),
                    "pred": int(yhat[i]),
                    "prob_c1": float(proba[i,0]),
                    "prob_c2": float(proba[i,1]),
                    "prob_c3": float(proba[i,2]),
                })

            # per-fold metrics
            acc = accuracy_score(yva, yhat)
            kap = cohen_kappa_score(yva, yhat, weights="quadratic")
            spr = spearmanr(yva, yhat)[0]; spr = 0.0 if np.isnan(spr) else float(spr)
            fold_rows.append({
                "repeat": rep, "fold": fold_id, "n_val": len(yva),
                "accuracy": float(acc), "qwk": float(kap), "spearman": float(spr),
            })

    # save OOF predictions & folds
    df_oof = pd.DataFrame(all_rows).sort_values(["index","repeat","fold"]).reset_index(drop=True)
    df_oof.to_csv(outdir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(outdir / "fold_metrics.csv", index=False)

    # aggregate across repeats by majority vote; average probabilities
    agg = (
        df_oof.groupby("index")
        .agg({
            "expert": "first",
            "pred": lambda s: _majority_vote_safe(s.astype(int).to_numpy()),
            "prob_c1": "mean", "prob_c2": "mean", "prob_c3": "mean",
            "case": "first"
        }).reset_index()
    )
    y_true = agg["expert"].to_numpy().astype(int)
    y_pred = agg["pred"].to_numpy().astype(int)
    proba = agg[["prob_c1","prob_c2","prob_c3"]].to_numpy().astype(float)

    # metrics + CI + p
    acc = accuracy_score(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    spr = spearmanr(y_true, y_pred)[0]; spr = 0.0 if np.isnan(spr) else float(spr)

    acc_lo, acc_hi = bootstrap_ci(accuracy_score, y_true, y_pred, n_boot=5000, seed=seed)
    kap_lo, kap_hi = bootstrap_ci(lambda a,b: cohen_kappa_score(a,b,weights="quadratic"),
                                  y_true, y_pred, n_boot=5000, seed=seed)
    spr_lo, spr_hi = bootstrap_ci(metric_spearman, y_true, y_pred, n_boot=5000, seed=seed)

    p_acc = perm_pvalue(accuracy_score, y_true, y_pred, n_perm=5000, seed=seed)
    p_kap = perm_pvalue(lambda a,b: cohen_kappa_score(a,b,weights="quadratic"), y_true, y_pred, n_perm=5000, seed=seed)
    p_spr = perm_pvalue(metric_spearman, y_true, y_pred, n_perm=5000, seed=seed)

    agg_row = {
        "denom": denom, "n_cases": len(y_true),
        "accuracy": acc, "accuracy_ci95_lo": acc_lo, "accuracy_ci95_hi": acc_hi, "p_acc_perm": p_acc,
        "qwk": kap, "qwk_ci95_lo": kap_lo, "qwk_ci95_hi": kap_hi, "p_qwk_perm": p_kap,
        "spearman": spr, "spearman_ci95_lo": spr_lo, "spearman_ci95_hi": spr_hi, "p_spr_perm": p_spr,
    }
    pd.DataFrame([agg_row]).to_csv(outdir / "aggregate_metrics.csv", index=False)

    # plots
    plot_confusions(y_true, y_pred, outdir, tag="oof")
    auprs = plot_pr_oof(y_true, proba, outdir, tag="oof")
    pd.DataFrame([{"class": k, "auprc": v} for k, v in auprs.items()]).to_csv(outdir / "auprc_oof.csv", index=False)

    # small readme
    with open(outdir / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Per-denominator OOF evaluation for denom={denom}.\n"
            f"Classifier={classifier}, CV={n_splits}-fold x {n_repeats} repeats.\n"
            "Features: single 1D per-case mean NDMI.\n"
        )

    return agg_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_method", type=str, default="flow_FB", choices=["flow_INR", "flow_FB", "Morph"])
    ap.add_argument("--classifier", type=str, default="svm", choices=["rf", "svm", "xgb"])
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict_case_match", type=int, default=0)  # 1 to disable base-id extraction
    ap.add_argument("--denoms", type=str, default="orig,clip,pct,mad")  # subset if needed
    args = ap.parse_args()

    IN_ROOT = Path(f"./NDMI_multiDen_{args.flow_method}")
    OUT_ROOT = Path(f"./agreement_per_denom_cv_{args.flow_method}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    STRICT = bool(args.strict_case_match)

    # load data
    df = load_dataset_means(IN_ROOT, STRICT)
    if df.empty:
        print("[ERROR] No cases found. Check inputs under", IN_ROOT)
        return

    # run for selected denoms
    denoms = [d.strip() for d in args.denoms.split(",") if d.strip() in DENOMS]
    if not denoms:
        denoms = DENOMS[:]

    summary_rows = []
    for d in denoms:
        row = run_one_denom(df, d, OUT_ROOT, args.classifier, args.n_splits, args.n_repeats, args.seed)
        summary_rows.append(row)

    # write aggregate summary across denoms
    pd.DataFrame(summary_rows).to_csv(OUT_ROOT / "aggregate_metrics_all_denoms.csv", index=False)
    print("[DONE] Outputs under:", OUT_ROOT)

if __name__ == "__main__":
    main()
