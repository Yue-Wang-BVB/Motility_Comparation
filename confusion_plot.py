import numpy as np
import matplotlib.pyplot as plt

cm_a = np.array([
    [61.7, 18.3, 20.0],
    [10.9, 70.3, 18.8],
    [0.0,   0.0, 100.0],
], dtype=float)

cm_b = np.array([
    [100.0,  0.0,  0.0],
    [  4.3, 62.3, 33.3],
    [  0.0, 12.5, 87.5],
], dtype=float)

cm_c = np.array([
    [95.7,  4.3,  0.0],
    [13.0, 87.0,  0.0],
    [ 6.5,  8.7, 84.8],
], dtype=float)

labels = ["1", "2", "3"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

cms = [cm_a, cm_b, cm_c]
subtitles = ["(a)", "(b)", "(c)"]

# 右边预留给 colorbar，这里先留得稍微宽一点
plt.subplots_adjust(wspace=0.2, right=0.88, bottom=0.18)

for ax, cm, sub in zip(axes, cms, subtitles):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)   # 不旋转
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    max_val = cm.max() if cm.size > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > max_val * 0.5 else "black"
            ax.text(
                j, i, f"{val:.1f}%",
                ha="center", va="center",
                color=color, fontsize=8
            )

    # (a)(b)(c) 放下面
    ax.text(
        0.5, -0.25, sub,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=10
    )

# ==== 这里是你要改的 legend/colorbar 部分 ====

# 宽度原来大概是 0.02，这里改成一半 0.01
# 高度原来是 0.7，这里改成 2/3 ≈ 0.47，并稍微往中间放一点
cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.47])
cbar = fig.colorbar(im, cax=cbar_ax)

# 不要整体的 % label
# cbar.ax.set_ylabel("%", rotation=90)  # 注释掉

# 每个 tick 后面加 %
ticks = cbar.get_ticks()
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.0f}%" for t in ticks])

plt.savefig("confu_replot.png", dpi=300, bbox_inches="tight")
plt.close()
