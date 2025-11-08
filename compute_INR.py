#!/usr/bin/env python3
"""
compute_INR.py  —  Per-pair 2-D INR registration (on-the-fly training)
=====================================================================
 • 输入 : test_sequences/healthy/sequences/<noise>/<case>/image.nii.gz
 • 输出 : results/flow_INR/<noise>/<case>/flow.npy
"""
from pathlib import Path
import numpy as np, nibabel as nib, csv, time, warnings, io
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm
import torch, pynvml, os

from data_preprocess import INR_models        # ← 你的 ImplicitRegistrator 实现

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────── 手动配置 ───────────────
NOISE_FILTER = None     # None 表示不过滤 "baseline"
CASE_FILTER  = "3-1"     # "6-2" None
START_FRAME  = 0             # 起始帧 (含)
END_FRAME    = None             # 结束帧 (含)；None = 最后一帧
# ───────────────────────────────────────

SEQ_ROOT = Path("./test_sequences/healthy/sequences")   # healthy
OUT_ROOT = Path("./results/flow_INR"); OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_CSV  = OUT_ROOT / f"runtime_log_inr_{CASE_FILTER}.csv"

# ---------- INR 超参 ----------
KWARGS = dict(
    verbose=False, network_type="SIREN", seed=42,
    # --- training ---
    epochs=1400, lr=1e-5, optimizer="adam", batch_size=16000,
    cycle_alpha=0.1, cycle_loss_schedule=True,
    # --- regularizers ---
    jacobian_regularization=False,  alpha_jacobian=0.005,
    bending_regularization=False,  alpha_bending=2e-4,
)

# ---------- 图像预处理 ----------
def preprocess(img):
    """不做 per-frame 归一化，仅确保 float32。"""
    return img.astype(np.float32)

# ---------- GPU util ----------
def reset_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pynvml.nvmlInit()

def gpu_stats():
    if torch.cuda.is_available():
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        mb   = torch.cuda.max_memory_allocated() / 1024**2
        return round(util, 1), round(mb, 1)
    return 0.0, 0.0

# ---------- register per pair ----------
def register_pair(img1, img2, mask1, mask2):
    """
    img1, img2 : (H,W) float32    — moving, fixed
    mask1, mask2 : (H,W) uint8    — 对应帧的分割掩膜 (0/1)
    return      : (H,W,2) float32 — 归一化位移 (dy_norm, dx_norm)
    """
    H, W = img1.shape
    moving = torch.tensor(img1[None, ...])
    fixed  = torch.tensor(img2[None, ...])

    kwargs = {**KWARGS,
              "mask":   mask1.astype(np.uint8),
              "mask_2": mask2.astype(np.uint8)}

    reg = INR_models.ImplicitRegistrator(moving, fixed, **kwargs)

    # ─── 静默内部 tqdm ───
    import tqdm as _tqdm_mod
    _orig_tqdm = _tqdm_mod.tqdm
    def _quiet(*a, **k):
        k["disable"] = True
        return _orig_tqdm(*a, **k)
    _tqdm_mod.tqdm = _quiet

    silent = io.StringIO()
    with redirect_stdout(silent), redirect_stderr(silent):
        reg.fit()                                   # ← 不再刷屏
    _tqdm_mod.tqdm = _orig_tqdm                     # 恢复

    flow = reg.predict_field((H, W), scale_to_pixel=False)                # (H,W,2) 像素单位
    return flow.astype(np.float32)

# ---------- collect sequences ----------
all_imgs = [p for p in SEQ_ROOT.rglob("image.nii.gz")]
sel_imgs = [p for p in all_imgs
            if (NOISE_FILTER is None or p.parts[-3] == NOISE_FILTER)
            and (CASE_FILTER  is None or p.parts[-2] == CASE_FILTER)]
print(f"[INFO] sequences to process: {len(sel_imgs)}")

# ---------- init CSV ----------
new_file = not LOG_CSV.exists()
with open(LOG_CSV, "a", newline="") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=["noise", "case", "pair_range",
                    "total_s", "s_per_pair", "gpu%", "MB_peak"])
    if new_file:
        writer.writeheader()

    for img_fp in sorted(sel_imgs):
        noise, case = img_fp.relative_to(SEQ_ROOT).parts[:2]
        out_dir = OUT_ROOT / noise / case; out_dir.mkdir(parents=True, exist_ok=True)
        flow_fp = out_dir / "flow.npy"
        if flow_fp.exists():
            print(f"[Skip] {noise}/{case}")
            continue

        # 读取序列
        vol = nib.load(str(img_fp)).get_fdata().astype(np.float32)
        # 主循环里预加载 entire mask volume
        mask_data = nib.load(str(SEQ_ROOT / noise / case / "label.nii.gz")).get_fdata().astype(np.uint8)

        # 整段统一归一化
        vmin, vmax = vol.min(), vol.max()
        if vmax - vmin < 1e-6:
            vol_norm = np.zeros_like(vol, np.float32)
        else:
            vol_norm = (vol - vmin) / (vmax - vmin)
        vol = vol_norm
        H, W = vol.shape[:2]

        s = max(START_FRAME, 0)
        e = vol.shape[-1] - 1 if END_FRAME is None else min(END_FRAME, vol.shape[-1] - 1)
        if e - s < 1:
            print(f"[WARN] {noise}/{case} frame range too small")
            continue
        vol = vol[..., s:e + 1]
        T = vol.shape[-1]

        flow_norm = np.zeros((*vol.shape[:2], T - 1, 2), np.float32)
        flow_pix = np.zeros_like(flow_norm)  # 用于存像素流

        prev = preprocess(vol[..., 0])
        reset_gpu(); t0 = time.time()

        for t in tqdm(range(0, T - 1), desc=f"{noise}/{case}", ncols=60, unit="frame"):
            img_t = preprocess(vol[..., t])
            img_tp1 = preprocess(vol[..., t + 1])
            m_t = mask_data[..., t] > 0
            m_tp1 = mask_data[..., t + 1] > 0

            v_norm = register_pair(img_t, img_tp1, m_t, m_tp1)

            # --------- 交换通道并保存归一化流 ---------
            v_norm_swapped = v_norm[..., [1, 0]]  # (dx, dy)
            flow_norm[..., t, :] = v_norm_swapped
            # --------- 转像素单位 (dx 用 W, dy 用 H) ---------
            v_pix = v_norm_swapped.copy()
            v_pix[..., 0] *= 0.5 * (W - 1)  # dx  ← 乘宽
            v_pix[..., 1] *= 0.5 * (H - 1)  # dy  ← 乘高
            flow_pix[..., t, :] = v_pix

        np.save(out_dir / "flow_norm.npy", flow_norm)  # 网络坐标 (-1,1)
        np.save(out_dir / "flow.npy", flow_pix)  # 像素坐标 (dx,dy)
        sec = time.time() - t0
        util, mb = gpu_stats()
        writer.writerow(dict(
            noise=noise, case=case, pair_range=f"{s}-{e - 1}",
            total_s=round(sec, 2), s_per_pair=round(sec / (T - 1), 3),
            **{"gpu%": util, "MB_peak": mb}))
        print(f"[Saved] flow_norm.npy {flow_norm.shape}  |  flow.npy {flow_pix.shape}")

print("[DONE] all INR flows saved →", OUT_ROOT)
