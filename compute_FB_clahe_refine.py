#!/usr/bin/env python3
"""
compute_FB_clahe.py  ·  Robust Farnebäck (global-norm)  ·  v2 with memory logging

- 明确区分并记录：
  (1) GPU 峰值显存（MB，PyTorch 计数器，peak）
  (2) CPU 峰值 RSS（MB，按序列采样得到的最大常驻内存）

输出 CSV：./results/flow_FB/runtime_log_fb_mem.csv
字段：noise, case, frames, time_s, gpu_util%, gpu_mem_peak_mb, cpu_rss_base_mb, cpu_rss_peak_mb
"""
import os, csv, time
from pathlib import Path
import cv2, nibabel as nib, numpy as np
from tqdm import tqdm

# ---- 新增依赖：psutil 用于 CPU RSS；pynvml/torch 用于 GPU 口径 ----
import psutil
import torch
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

# ===== 路径设置 =====
ROOT      = Path("./test_sequences/healthy/sequences")
FLOW_ROOT = Path("./results_refine/flow_FB")
FLOW_ROOT.mkdir(parents=True, exist_ok=True)

# 新日志名，避免与旧表头冲突；若要复用旧文件名，将下一行改为 "runtime_log_fb.csv"
LOG_CSV   = FLOW_ROOT / "runtime_log_fb_mem.csv"

# ===== CLAHE & 归一化 =====
CLAHE_ON   = True                             # False 可完全关闭直方图均衡
CLAHE      = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

def build_normalizer(vol: np.ndarray):
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-6:
        return lambda x: np.zeros_like(x, np.uint8)
    k = 255.0 / (vmax - vmin)
    return lambda x: np.clip((x - vmin) * k, 0, 255).astype("uint8")

def prep_frame(frame_f32: np.ndarray, to_u8):
    img_u8 = to_u8(np.squeeze(frame_f32))
    if CLAHE_ON:
        img_u8 = CLAHE.apply(img_u8)
    return img_u8

# ====== GPU 工具（更健壮）======
def reset_gpu():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if _HAS_NVML:
                pynvml.nvmlInit()
        except Exception:
            pass  # 保底不影响 CPU 路径

def gpu_stats():
    util, mb = 0.0, 0.0
    if torch.cuda.is_available():
        try:
            if _HAS_NVML:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            mb = float(torch.cuda.max_memory_allocated()) / 1024**2
        except Exception:
            pass
    return round(util, 1), round(mb, 1)

# ====== CPU RSS 采样工具（每序列取峰值）======
_PROC = psutil.Process(os.getpid())
def _rss_mb():
    try:
        return _PROC.memory_info().rss / (1024**2)
    except Exception:
        return 0.0

# ===== Farnebäck 参数 =====
FB_KW = dict(
    pyr_scale=0.3,
    levels=5,
    winsize=21,
    iterations=3,
    poly_n=7,
    poly_sigma=1.5,
    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
)

# ===== 初始化日志 =====
FIELDNAMES = [
    "noise", "case", "frames", "time_s",
    "gpu_util%", "gpu_mem_peak_mb",
    "cpu_rss_base_mb", "cpu_rss_peak_mb",
]
new_file = not LOG_CSV.exists()
with open(LOG_CSV, "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
    if new_file:
        writer.writeheader()

# ===== 遍历所有 image.nii.gz =====
image_files = sorted(ROOT.rglob("image.nii.gz"))
print(f"[INFO] Found {len(image_files)} sequences")

for img_path in tqdm(image_files, desc="Robust FB"):
    noise, case = img_path.relative_to(ROOT).parts[:2]
    out_dir = FLOW_ROOT / noise / case
    flow_path = out_dir / "flow.npy"
    if flow_path.exists():
        tqdm.write(f"[Skip] {noise}/{case} already exists")
        continue
    out_dir.mkdir(parents=True, exist_ok=True)

    reset_gpu()
    t0 = time.time()

    # ---- CPU 内存：基线与峰值（按序列采样）----
    cpu_base = _rss_mb()
    cpu_peak = cpu_base

    # 1) 读取序列并创建归一化器
    vol = nib.load(str(img_path)).get_fdata().astype(np.float32)  # (H,W,T)
    H, W, T = vol.shape
    cpu_peak = max(cpu_peak, _rss_mb())

    norm = build_normalizer(vol)

    # 2) 预分配光流数组（大块内存）
    flow = np.zeros((H, W, T - 1, 2), np.float32)
    cpu_peak = max(cpu_peak, _rss_mb())

    # 3) 逐帧计算光流（关键步骤内亦采样）
    prev = prep_frame(vol[:, :, 0], norm)
    cpu_peak = max(cpu_peak, _rss_mb())

    for t in range(1, T):
        curr = prep_frame(vol[:, :, t], norm)
        vec = cv2.calcOpticalFlowFarneback(prev, curr, None, **FB_KW)
        # 若你需要中位数滤波，可保留下一行；否则可注释以减少额外内存抖动
        vec = cv2.medianBlur(vec, 3)
        flow[:, :, t - 1, :] = vec
        prev = curr

        # 每帧一次采样即可（开销极小）
        if t % 1 == 0:
            cpu_peak = max(cpu_peak, _rss_mb())

    # 4) 保存结果并记录一次
    np.save(flow_path, flow)
    cpu_peak = max(cpu_peak, _rss_mb())

    sec = time.time() - t0
    gpu_util, gpu_mb = gpu_stats()
    tqdm.write(
        f"[Saved] {noise}/{case} | {T-1} flows | {sec:.2f}s | "
        f"GPU {gpu_util:.1f}%/{gpu_mb:.1f}MB | CPU RSS peak {cpu_peak:.2f}MB"
    )

    with open(LOG_CSV, "a", newline="") as fh_log:
        writer = csv.DictWriter(fh_log, fieldnames=FIELDNAMES)
        writer.writerow(dict(
            noise=noise, case=case, frames=T-1, time_s=round(sec, 2),
            **{"gpu_util%": gpu_util, "gpu_mem_peak_mb": gpu_mb},
            cpu_rss_base_mb=round(cpu_base, 2),
            cpu_rss_peak_mb=round(cpu_peak, 2),
        ))
