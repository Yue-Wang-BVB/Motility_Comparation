import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# === Configuration Parameters ===
CASE_FILTER  = ["1"]                 # [] = all cases, or specify e.g. ["11-1", "1-3"]
NOISE_FILTER = ["baseline"]         # [] = all noise types, or specify e.g. ["baseline", "gauss_sigma030"]
START_FRAME  = 0                    # Starting frame index (inclusive)
END_FRAME_EXC = 50                  # Ending frame index (exclusive), e.g. 50
FLOW_METHOD  = "flow_FB"            # Choose from flow_INR / flow_FB
ROOT         = Path("./noisy_sequences")               # Root directory for original images & masks
FLOW_ROOT    = Path("./results") / FLOW_METHOD         # Root directory for flow.npy
SAVE_ROOT    = Path("./visualization") / FLOW_METHOD   # Root directory to save arrow plots
STEP  = 2        # Arrow grid stride
SCALE = 1        # Arrow scaling
WIDTH = 0.0025   # Arrow line width

# === Optical Flow Visualization Function ===
def visualize_flow_arrow(image_t, flow, mask_t=None, step=6, scale=1, width=0.0025, save_path=None, title=None):
    h, w = image_t.shape
    y, x = np.mgrid[0:h:step, 0:w:step]
    u = flow[::step, ::step, 0]  # dx → horizontal motion
    v = flow[::step, ::step, 1]  # dy → vertical motion

    # Apply mask if provided
    if mask_t is not None:
        mask_valid = mask_t[::step, ::step] > 0
        x = x[mask_valid]; y = y[mask_valid]
        u = u[mask_valid]; v = v[mask_valid]

    # Normalize image for display
    img_norm = (image_t - image_t.min()) / (image_t.max() - image_t.min() + 1e-8)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_norm, cmap='gray')
    plt.quiver(x, y, u, v, color='cyan', angles='xy', scale_units='xy', scale=scale, width=width)
    plt.title(title or "Optical Flow Arrows")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# === Traverse all flow.npy files ===
flow_paths = list(FLOW_ROOT.rglob("flow.npy"))
print(f"[INFO] Found {len(flow_paths)} flow fields.")

for flow_path in sorted(flow_paths):
    rel = flow_path.relative_to(FLOW_ROOT)
    noise, case = rel.parts[0], rel.parts[1]

    # 1) Filter by noise type
    if NOISE_FILTER and noise not in NOISE_FILTER:
        continue
    # 2) Filter by case ID
    if CASE_FILTER and case not in CASE_FILTER:
        continue

    image_path = ROOT / noise / case / "image.nii.gz"
    mask_path  = ROOT / noise / case / "label.nii.gz"
    save_dir   = SAVE_ROOT / noise / case
    os.makedirs(save_dir, exist_ok=True)

    # Check existence
    if not image_path.exists() or not mask_path.exists():
        print(f"[WARN] Missing image or mask for {noise}/{case}, skipped.")
        continue

    print(f"Processing {noise}/{case}…")

    # Load image, mask, and flow data
    img_data  = nib.load(str(image_path)).get_fdata().astype(np.float32)
    mask_data = nib.load(str(mask_path)).get_fdata().astype(np.uint8)
    flow_data = np.load(str(flow_path)).astype(np.float32)   # shape: (H, W, T-1, 2)
    H, W, T = img_data.shape

    # Set time range to visualize
    start = max(START_FRAME, 0)
    end   = min(END_FRAME_EXC, T - 1)

    # Generate visualization for each frame t → t+1
    for t in range(start, end):
        flow_idx = t - start
        if flow_idx >= flow_data.shape[2]:
            break
        frame   = img_data[..., t]
        flow    = flow_data[..., flow_idx, :]
        mask_t1 = mask_data[..., t]

        save_path = save_dir / f"t{t:02d}.png"
        title = f"{noise}/{case} Flow t={t}→t+1"
        visualize_flow_arrow(frame, flow, mask_t=mask_t1,
                             step=STEP, scale=SCALE, width=WIDTH,
                             save_path=str(save_path), title=title)
    print(f"Saved arrows: {noise}/{case}")
