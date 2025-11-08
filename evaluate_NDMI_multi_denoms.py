#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_MI_jacobian_NDMI_multi_denoms.py
=========================================
Compute NDMI curves with FOUR denominator choices, following the original logic.

Now supports three METHODs:
  - METHOD="flow_INR": use ./results/flow_INR/<noise>/<case>/flow.npy + GT ROI
  - METHOD="flow_FB" : use ./results/flow_FB/<noise>/<case>/flow.npy  + GT ROI
  - METHOD="Morph"   : use ./results/seg/<noise>/<case>/filtered_mask.nii.gz (if present;
                       otherwise fall back to GT label) and define F_t as the **area**
                       (voxel count) of the mask at frame t.

Denominator variants:
  1) 'orig' : per-frame scale = |F_t| + EPS
  2) 'clip' : per-frame scale = max(|F_t|, tau_clip), where tau_clip is the P_FLOOR percentile
              of |F| across the case (for Morph) or |J_t| over ROI_t (for flow)
  3) 'pct'  : per-frame scale = s_pct, a robust case-scale (median |F|) (scalar)
  4) 'mad'  : per-frame scale = s_mad, robust MAD(|F|) (scalar)

Saved outputs (per case):
  - NDMI curves: <case>_NDMI-<denom>.npy (after SKIP_FRM)
  - Scales used: <case>_scales.npz  (arrays per frame: tau_clip, s_pct, s_mad; meta)
  - Summary CSV with per-case statistics for each denominator.

Directory structure / I/O:
  - ROI label (GT): ./test_sequences/healthy/sequences/<noise>/<case>/label.nii.gz
  - Flow:          ./results/<METHOD>/<noise>/<case>/flow.npy              (flow_* only)
  - Seg (pred):    ./results/seg/<noise>/<case>/filtered_mask.nii.gz       (Morph only)
  - Outputs:       ./NDMI_multiDen_<METHOD>/**

Dependencies: numpy · nibabel · cv2 · pandas · tqdm
"""

from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
import pandas as pd
from tqdm import tqdm

# ------------------- Config -------------------
METHOD       = "Morph"      # "flow_INR", "flow_FB", or "Morph"
FLOW_ROOT    = Path("./results") / METHOD  # used only for flow_* methods
SEG_ROOT     = Path("./results/seg")       # used only for Morph
DATA_ROOT    = Path("./test_sequences/healthy/sequences")
SAVE_ROOT    = Path(f"./NDMI_multiDen_{METHOD}")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

SIGMA        = 0.4          # Gaussian sigma for smoothing flow
SKIP_FRM     = 3            # skip the first N NDMI points
EPS          = 1e-6         # small epsilon
P_FLOOR      = 1.0          # percentile p for tau_clip (e.g., 1.0 -> 1st percentile)
MAD_CONST    = 1.4826       # to make MAD comparable to std under Gaussian
NOISE_FILTER = []           # e.g., ["baseline"] or [] for all
CASE_FILTER  = []           # e.g., ["1-1"] or [] for all

# ------------------- Helpers -------------------
def jac_det(flow_xy: np.ndarray) -> np.ndarray:
    """Compute Jacobian determinant det(∇u + I) for a flow field (H,W,2)."""
    fx_x, fx_y = np.gradient(flow_xy[..., 0])  # dux/dx, dux/dy
    fy_x, fy_y = np.gradient(flow_xy[..., 1])  # duy/dx, duy/dy
    J = (1.0 + fx_x) * (1.0 + fy_y) - fx_y * fy_x
    return J.astype(np.float32)

def gaussian_smooth_flow(flow: np.ndarray, sigma: float) -> np.ndarray:
    """Apply isotropic Gaussian smoothing to flow array (H, W, T-1, 2)."""
    H, W, Tm1, _ = flow.shape
    sm = cv2.GaussianBlur(flow.reshape(-1, 2), (0, 0), sigma)
    return sm.reshape(H, W, Tm1, 2)

def robust_mad(arr: np.ndarray) -> float:
    """Median absolute deviation (scalar)."""
    med = np.median(arr)
    return float(MAD_CONST * np.median(np.abs(arr - med)))

def compute_area_series(mask3d: np.ndarray) -> np.ndarray:
    """Return per-frame area (voxel count) series from a 3D mask (H,W,T)."""
    return (mask3d > 0).sum(axis=(0, 1)).astype(np.float64)

# ------------------- Main -------------------
def main():
    rows = []

    if METHOD.startswith("flow_"):
        flow_paths = list(FLOW_ROOT.rglob("flow.npy"))
        print(f"[INFO] METHOD={METHOD}. Found {len(flow_paths)} flow files under {FLOW_ROOT}")
        for fp in tqdm(sorted(flow_paths), desc="NDMI multi-denominators (flow)", unit="case"):
            # Expect: ./results/<FLOW_METHOD>/<noise>/<case>/flow.npy
            try:
                noise, case = fp.relative_to(FLOW_ROOT).parts[:2]
            except Exception:
                tqdm.write(f"[WARN] Unexpected path structure: {fp}")
                continue
            if NOISE_FILTER and noise not in NOISE_FILTER: continue
            if CASE_FILTER and case not in CASE_FILTER:     continue

            mask_path = DATA_ROOT / noise / case / "label.nii.gz"
            if not mask_path.exists():
                tqdm.write(f"[WARN] Missing mask for {noise}/{case}")
                continue

            try:
                flow = np.load(fp)  # (H, W, T-1, 2)
                mask = nib.load(str(mask_path)).get_fdata().astype(np.uint8)  # (H, W, T)
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to load {noise}/{case}: {e}")
                continue

            H, W, Tm1 = flow.shape[:3]
            if mask.shape[0] != H or mask.shape[1] != W:
                tqdm.write(f"[WARN] Size mismatch for {noise}/{case}: flow {flow.shape}, mask {mask.shape}")
                continue

            # Smooth and compute per-frame Jacobian determinants
            flow_sm = gaussian_smooth_flow(flow, SIGMA)
            J_list = [jac_det(flow_sm[..., t, :]) for t in range(Tm1)]  # t = 0..Tm1-1

            # Prepare arrays to collect per-frame scales (for saving)
            tau_clip_all, s_pct_all, s_mad_all = [], [], []

            # NDMI curves per denominator
            mi_orig, mi_clip, mi_pct, mi_mad = [], [], [], []

            # Iterate over consecutive frame pairs (t -> t+1), as in original
            for t in range(Tm1 - 1):
                # ROI uses mask[..., t+1] > 0 (to mirror original)
                roi = (mask[..., t + 1] > 0)
                if not np.any(roi):
                    tau_clip_all.append(0.0); s_pct_all.append(0.0); s_mad_all.append(0.0)
                    mi_orig.append(0.0); mi_clip.append(0.0); mi_pct.append(0.0); mi_mad.append(0.0)
                    continue

                Jt   = J_list[t][roi]
                Jt1  = J_list[t + 1][roi]
                dJ   = np.abs(Jt1 - Jt)
                aJt  = np.abs(Jt)

                # Per-frame robust scales and floors (computed over ROI voxels at frame t)
                tau_clip = float(np.percentile(aJt, P_FLOOR))
                s_pct    = float(np.median(aJt))
                s_mad    = float(robust_mad(aJt))

                if s_pct <= 0: s_pct = max(tau_clip, EPS)
                if s_mad <= 0: s_mad = max(tau_clip, EPS)

                tau_clip_all.append(tau_clip)
                s_pct_all.append(s_pct)
                s_mad_all.append(s_mad)

                # 1) original
                denom_orig = aJt + EPS
                mi_orig.append(float(np.mean(dJ / denom_orig)))

                # 2) clipping floor on |J_t|
                denom_clip = np.maximum(aJt, tau_clip)
                mi_clip.append(float(np.mean(dJ / denom_clip)))

                # 3) percentile-based (median) scale, per-frame scalar
                mi_pct.append(float(np.mean(dJ / s_pct)))

                # 4) MAD-based scale, per-frame scalar
                mi_mad.append(float(np.mean(dJ / s_mad)))

            # Apply SKIP_FRM to curves and scales
            mi_orig = mi_orig[SKIP_FRM:]
            mi_clip = mi_clip[SKIP_FRM:]
            mi_pct  = mi_pct[SKIP_FRM:]
            mi_mad  = mi_mad[SKIP_FRM:]
            tau_clip_all = tau_clip_all[SKIP_FRM:]
            s_pct_all    = s_pct_all[SKIP_FRM:]
            s_mad_all    = s_mad_all[SKIP_FRM:]

            # Save per-case curves
            out_dir = SAVE_ROOT / noise
            out_dir.mkdir(exist_ok=True, parents=True)
            np.save(out_dir / f"{case}_NDMI-orig.npy", np.array(mi_orig, dtype=np.float32))
            np.save(out_dir / f"{case}_NDMI-clip.npy", np.array(mi_clip, dtype=np.float32))
            np.save(out_dir / f"{case}_NDMI-pct.npy",  np.array(mi_pct,  dtype=np.float32))
            np.save(out_dir / f"{case}_NDMI-mad.npy",  np.array(mi_mad,  dtype=np.float32))

            # Save per-frame scales used
            np.savez_compressed(out_dir / f"{case}_scales.npz",
                                tau_clip=np.array(tau_clip_all, dtype=np.float32),
                                s_pct=np.array(s_pct_all, dtype=np.float32),
                                s_mad=np.array(s_mad_all, dtype=np.float32),
                                meta=np.array([SIGMA, SKIP_FRM, EPS, P_FLOOR], dtype=np.float32))

            # Append per-case summary
            def _stats(arr):
                return (float(np.mean(arr)) if len(arr) else 0.0,
                        float(np.std(arr))  if len(arr) else 0.0,
                        float(np.max(arr))  if len(arr) else 0.0)

            m1, s1, x1 = _stats(mi_orig)
            m2, s2, x2 = _stats(mi_clip)
            m3, s3, x3 = _stats(mi_pct)
            m4, s4, x4 = _stats(mi_mad)

            rows.append({
                "noise": noise, "case": case, "frames": len(mi_orig),
                "orig_mean": m1, "orig_std": s1, "orig_max": x1,
                "clip_mean": m2, "clip_std": s2, "clip_max": x2,
                "pct_mean":  m3, "pct_std":  s3, "pct_max":  x3,
                "mad_mean":  m4, "mad_std":  s4, "mad_max":  x4,
            })

    elif METHOD == "Morph":
        noise_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
        print(f"[INFO] METHOD=Morph. Scanning {len(noise_dirs)} noise dirs under {DATA_ROOT}")
        for noise_dir in tqdm(noise_dirs, desc="NDMI multi-denominators (Morph)", unit="noise"):
            noise = noise_dir.name
            if NOISE_FILTER and noise not in NOISE_FILTER: continue
            case_dirs = sorted([d for d in (DATA_ROOT / noise).iterdir() if d.is_dir()])
            for case_dir in case_dirs:
                case = case_dir.name
                if CASE_FILTER and case not in CASE_FILTER: continue

                gt_path   = DATA_ROOT / noise / case / "label.nii.gz"
                pred_path = SEG_ROOT / noise / case / "filtered_mask.nii.gz"
                if not gt_path.exists():
                    tqdm.write(f"[WARN] Missing GT label for {noise}/{case}")
                    continue
                try:
                    mask_gt = nib.load(str(gt_path)).get_fdata().astype(np.uint8)  # (H,W,T)
                    if pred_path.exists():
                        mask_pred = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
                        mask_use = mask_pred
                    else:
                        mask_use = mask_gt
                except Exception as e:
                    tqdm.write(f"[ERROR] Failed to load masks for {noise}/{case}: {e}")
                    continue

                A = compute_area_series(mask_use)  # (T,)
                if A.size < 2:
                    tqdm.write(f"[WARN] Too few frames for {noise}/{case}")
                    continue

                absA = np.abs(A)
                s_pct_case = float(np.median(absA))
                s_mad_case = float(robust_mad(absA))
                if s_pct_case <= 0: s_pct_case = float(np.quantile(absA, P_FLOOR/100.0))
                if s_mad_case <= 0: s_mad_case = float(np.quantile(absA, P_FLOOR/100.0))

                tau_clip_case = float(np.percentile(absA, P_FLOOR))

                mi_orig, mi_clip, mi_pct, mi_mad = [], [], [], []
                tau_clip_all, s_pct_all, s_mad_all = [], [], []
                for t in range(len(A) - 1):
                    num = float(abs(A[t+1] - A[t]))
                    denom_orig = float(absA[t] + EPS)
                    denom_clip = float(max(absA[t], tau_clip_case))
                    denom_pct  = float(s_pct_case)
                    denom_mad  = float(s_mad_case if s_mad_case > 0 else denom_pct)

                    mi_orig.append(num / denom_orig)
                    mi_clip.append(num / denom_clip)
                    mi_pct.append(num / denom_pct)
                    mi_mad.append(num / denom_mad)

                    tau_clip_all.append(tau_clip_case)
                    s_pct_all.append(s_pct_case)
                    s_mad_all.append(denom_mad)

                # Apply SKIP_FRM
                mi_orig = mi_orig[SKIP_FRM:]
                mi_clip = mi_clip[SKIP_FRM:]
                mi_pct  = mi_pct[SKIP_FRM:]
                mi_mad  = mi_mad[SKIP_FRM:]
                tau_clip_all = tau_clip_all[SKIP_FRM:]
                s_pct_all    = s_pct_all[SKIP_FRM:]
                s_mad_all    = s_mad_all[SKIP_FRM:]

                out_dir = SAVE_ROOT / noise
                out_dir.mkdir(exist_ok=True, parents=True)
                np.save(out_dir / f"{case}_NDMI-orig.npy", np.array(mi_orig, dtype=np.float32))
                np.save(out_dir / f"{case}_NDMI-clip.npy", np.array(mi_clip, dtype=np.float32))
                np.save(out_dir / f"{case}_NDMI-pct.npy",  np.array(mi_pct,  dtype=np.float32))
                np.save(out_dir / f"{case}_NDMI-mad.npy",  np.array(mi_mad,  dtype=np.float32))

                np.savez_compressed(out_dir / f"{case}_scales.npz",
                                    tau_clip=np.array(tau_clip_all, dtype=np.float32),
                                    s_pct=np.array(s_pct_all, dtype=np.float32),
                                    s_mad=np.array(s_mad_all, dtype=np.float32),
                                    meta=np.array([SKIP_FRM, EPS, P_FLOOR], dtype=np.float32))

                def _stats(arr):
                    return (float(np.mean(arr)) if len(arr) else 0.0,
                            float(np.std(arr))  if len(arr) else 0.0,
                            float(np.max(arr))  if len(arr) else 0.0)

                m1, s1, x1 = _stats(mi_orig)
                m2, s2, x2 = _stats(mi_clip)
                m3, s3, x3 = _stats(mi_pct)
                m4, s4, x4 = _stats(mi_mad)

                rows.append({
                    "noise": noise, "case": case, "frames": len(mi_orig),
                    "orig_mean": m1, "orig_std": s1, "orig_max": x1,
                    "clip_mean": m2, "clip_std": s2, "clip_max": x2,
                    "pct_mean":  m3, "pct_std":  s3, "pct_max":  x3,
                    "mad_mean":  m4, "mad_std":  s4, "mad_max":  x4,
                })
    else:
        raise ValueError(f"Unknown METHOD={METHOD}; choose 'flow_INR', 'flow_FB', or 'Morph'.")

    # Save global summary
    if rows:
        df = pd.DataFrame(rows).sort_values(["noise", "case"]).reset_index(drop=True)
        df.to_csv(SAVE_ROOT / "summary.csv", index=False)
        print("[DONE] Summary saved →", SAVE_ROOT / "summary.csv")
    else:
        print("[DONE] No valid cases processed.")

if __name__ == "__main__":
    main()
