#!/usr/bin/env python3
"""
generate_noise_sequences.py
===========================
Batch-generate single-factor noisy cine-MRI sequences (intensity drift, Gaussian noise, affine distortion)
for robustness experiments on small-bowel motility.

Input layout (under --root_dir, default: ./example_data):
└── example_data
    ├── 1/                # case id (any folder name)
    │   ├── image.nii.gz
    │   └── label.nii.gz
    ├── 2/
    └── …

Output layout (created automatically):
└── noisy_sequences/
    ├── baseline/
    │   └── 1/ (image.nii.gz, label.nii.gz, meta.json)
    ├── drift_k110_b+05/
    │   └── 1/
    ├── gauss_sigma030/
    └── affine_lvl2/

Each output case additionally contains meta.json that records the noise parameters.

Dependencies: numpy, scipy, nibabel.
"""
import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform

# -----------------------------------------------------------------------------
# NOISE CONFIGURATIONS (adjustable)
# -----------------------------------------------------------------------------
DRIFT_KAPPAS = [0.90, 0.95, 1.00, 1.05, 1.10]       # ±10%, ±5%, baseline
DRIFT_BETAS  = [-0.05, 0.0, 0.05]                   # ±5%, 0 additive offset

GAUSS_SIGMAS = [0.01, 0.03, 0.05, 0.07]             # noise std as fraction of [0, 1]

# Affine distortions at 3 severity levels (scale + shear + translation)
AFFINE_LEVELS = {
    "lvl1": dict(scale=1.02, shear_deg=0.5, tx=1,  ty=0),
    "lvl2": dict(scale=1.04, shear_deg=1.0, tx=2,  ty=0),
    "lvl3": dict(scale=0.98, shear_deg=-1.0, tx=-2, ty=0),
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def load_nifti(path: Path):
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32), img.affine

def normalize01(vol: np.ndarray):
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin == 0:
        return vol * 0.0
    return (vol - vmin) / (vmax - vmin)

def save_case(out_dir: Path, image: np.ndarray, mask: np.ndarray, affine, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(image.astype(np.float32), affine), out_dir / "image.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine),  out_dir / "label.nii.gz")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

# Noise injection functions

def apply_drift(vol: np.ndarray, kappa: float, beta: float):
    return np.clip(kappa * vol + beta, 0.0, 1.0)

def apply_gaussian(vol: np.ndarray, sigma: float):
    noise = np.random.normal(0.0, sigma, size=vol.shape).astype(np.float32)
    return np.clip(vol + noise, 0.0, 1.0)

def affine_params_to_matrix(scale: float, shear_deg: float, tx: float, ty: float):
    shear = np.tan(np.deg2rad(shear_deg))
    mat = np.array([[scale, shear, tx],
                    [0.0,   scale, ty],
                    [0.0,   0.0,   1.0]], dtype=np.float32)
    return mat

def apply_affine(vol: np.ndarray, mask: np.ndarray, scale: float, shear_deg: float, tx: float, ty: float):
    A = affine_params_to_matrix(scale, shear_deg, tx, ty)
    invA = np.linalg.inv(A)

    warped_vol  = np.zeros_like(vol, dtype=np.float32)
    warped_mask = np.zeros_like(mask, dtype=np.uint8)

    for t in range(vol.shape[-1]):
        warped_vol[..., t] = affine_transform(vol[..., t], invA[:2, :2], offset=invA[:2, 2],
                                              order=1, mode="nearest")
        warped_mask[..., t] = affine_transform(mask[..., t], invA[:2, :2], offset=invA[:2, 2],
                                               order=0, mode="nearest")

    return warped_vol, warped_mask, A

# -----------------------------------------------------------------------------
# MAIN PROCESSING FUNCTION FOR EACH CASE
# -----------------------------------------------------------------------------

def process_case(case_dir: Path, noisy_root: Path):
    img_path  = case_dir / "image.nii.gz"
    mask_path = case_dir / "label.nii.gz"
    img, affine = load_nifti(img_path)
    mask, _     = load_nifti(mask_path)

    img = normalize01(img)

    # Save baseline (original normalized image)
    baseline_dir = noisy_root / "baseline" / case_dir.name
    save_case(baseline_dir, img, mask, affine, {"type": "baseline"})

    # Intensity drift noise
    for kappa in DRIFT_KAPPAS:
        for beta in DRIFT_BETAS:
            noise_name = f"drift_k{int(kappa*100):03d}_b{int(beta*100):+03d}"
            out_dir = noisy_root / noise_name / case_dir.name
            noisy_img = apply_drift(img, kappa, beta)
            save_case(out_dir, noisy_img, mask, affine,
                      {"type": "drift", "kappa": kappa, "beta": beta})

    # Gaussian noise
    for sigma in GAUSS_SIGMAS:
        noise_name = f"gauss_sigma{int(sigma*1000):03d}"
        out_dir = noisy_root / noise_name / case_dir.name
        noisy_img = apply_gaussian(img, sigma)
        save_case(out_dir, noisy_img, mask, affine,
                  {"type": "gauss", "sigma": sigma})

    # Affine distortion
    for lvl, p in AFFINE_LEVELS.items():
        noise_name = f"affine_{lvl}"
        out_dir = noisy_root / noise_name / case_dir.name
        noisy_img, noisy_mask, A = apply_affine(img, mask, **p)
        save_case(out_dir, noisy_img, noisy_mask, affine,
                  {"type": "affine", **p, "matrix": A.tolist()})

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate noisy cine-MRI sequences")
    parser.add_argument("--root_dir", type=str, default="./example_data",
                        help="Directory containing subfolders with cases")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed for Gaussian noise generation")
    args = parser.parse_args()

    np.random.seed(args.seed)

    root = Path(args.root_dir)
    noisy_root = root.parent / "noisy_sequences"
    noisy_root.mkdir(exist_ok=True)

    # Discover case directories
    case_dirs = [d for d in root.iterdir() if d.is_dir()]
    case_dirs.sort(key=lambda p: int(re.match(r"\d+", p.name).group()) if re.match(r"\d+", p.name) else p.name)

    print(f"[INFO] Found {len(case_dirs)} cases. Generating noisy variants...")
    for idx, case_dir in enumerate(case_dirs, 1):
        print(f"  ({idx}/{len(case_dirs)}) case {case_dir.name}")
        process_case(case_dir, noisy_root)

    print("[DONE] All noisy sequences saved to:", noisy_root)

if __name__ == "__main__":
    main()
