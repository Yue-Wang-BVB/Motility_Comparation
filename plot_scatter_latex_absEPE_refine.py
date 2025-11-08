#!/usr/bin/env python3
"""
Plot combined scatter for flow_FB and flow_INR

- 读取 ../MI_Jdet_sigma_sweep_<METHOD>/sigma<σ>/summary.csv
- 计算 ΔNDMI% (百分比) 和 ΔEPE (绝对像素差, pixels)
- 聚合为大类噪声 (affine / gauss / drift) → 散点图
- 在同一张图中上下排列两个子图：
    (a) flow_FB
    (b) flow_INR
- 提升打印可读性：
    * 明确坐标轴单位
    * 各自使用合适的坐标范围与刻度
    * 虚线规则：
        - 上图: x 轴每 10、一条虚线；y 轴每 0.1 一条虚线
        - 下图: x 轴每 20、一条虚线；y 轴每 1   一条虚线
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------- 全局参数 --------
SIGMA = 0.50                  # 目标 σ
EPS_MI = 1e-8                 # 避免除 0
METHODS = ["flow_FB", "flow_INR"]

sns.set_style("whitegrid")


def load_delta_table(flow_method: str):
    """读取指定方法的数据并计算 ΔNDMI% 和 ΔEPE."""
    root = Path(f"../MI_Jdet_sigma_sweep_{flow_method}")
    src_csv = root / f"sigma{SIGMA:.2f}" / "summary.csv"
    if not src_csv.exists():
        raise FileNotFoundError(f"{src_csv} 不存在，请先运行 {flow_method} 的 sigma 扫描脚本。")

    df = pd.read_csv(src_csv)

    # baseline
    base = df[df.noise == "baseline"].set_index("case")[["MI", "EPE"]]
    base = base.rename(columns={"MI": "MI_base", "EPE": "EPE_base"})

    records = []
    for noise in df.noise.unique():
        if noise == "baseline":
            continue
        cur = df[df.noise == noise].set_index("case")
        common = base.index.intersection(cur.index)
        if common.empty:
            continue

        delta_mi = ((cur.loc[common, "MI"] - base.loc[common, "MI_base"]) /
                    (base.loc[common, "MI_base"].replace(0, EPS_MI))) * 100.0
        delta_epe = (cur.loc[common, "EPE"] - base.loc[common, "EPE_base"])

        for c in common:
            records.append({
                "case": c,
                "noise": noise,
                "ΔNDMI%": delta_mi.loc[c],
                "ΔEPE": delta_epe.loc[c]
            })

    plot_df = pd.DataFrame(records).dropna()

    # 噪声大类标签
    def group_tag(n):
        if isinstance(n, str):
            if n.startswith("gauss"):
                return "gauss"
            if n.startswith("affine"):
                return "affine"
            if n.startswith("drift"):
                return "drift"
        return n

    plot_df["group"] = plot_df["noise"].map(group_tag)

    # 聚合为 (group, noise) 平均值
    plot_mean = (
        plot_df.groupby(["group", "noise"])
        .agg({"ΔNDMI%": "mean", "ΔEPE": "mean"})
        .reset_index()
    )

    return plot_mean, root


def nice_limits(values, padding_ratio=0.08):
    """根据数据范围生成稍微留白的轴范围."""
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return -1.0, 1.0
    if np.isclose(vmin, vmax):
        span = abs(vmax) if vmax != 0 else 1.0
        return -span, span
    span = vmax - vmin
    pad = span * padding_ratio
    return vmin - pad, vmax + pad


def nice_ticks(vmin, vmax, n=5):
    """生成较稀疏刻度."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        return [vmin, vmax]
    return list(np.linspace(vmin, vmax, n))


def guide_lines(ax, x_min, x_max, y_min, y_max, idx):
    """
    添加浅灰虚线参考线:
    - 上图 (idx=0):
        x: 每 10
        y: 每 0.1
    - 下图 (idx=1):
        x: 每 20
        y: 每 1
    跳过 0，0 由单独的加深参考线表示。
    """
    if idx == 0:
        x_step = 10.0
        y_step = 0.1
    else:
        x_step = 20.0
        y_step = 1.0

    # x 轴虚线
    if x_step > 0:
        start_x = np.ceil(x_min / x_step) * x_step
        end_x = np.floor(x_max / x_step) * x_step
        xs = np.arange(start_x, end_x + 1e-8, x_step)
        xs = np.round(xs, 6)
        for xv in xs:
            if abs(xv) < 1e-8:
                continue
            ax.axvline(
                xv,
                ls="--",
                lw=0.7,
                color="gray",
                alpha=0.35,
                zorder=0
            )

    # y 轴虚线
    if y_step > 0:
        start_y = np.ceil(y_min / y_step) * y_step
        end_y = np.floor(y_max / y_step) * y_step
        ys = np.arange(start_y, end_y + 1e-8, y_step)
        ys = np.round(ys, 6)
        for yv in ys:
            if abs(yv) < 1e-8:
                continue
            ax.axhline(
                yv,
                ls="--",
                lw=0.7,
                color="gray",
                alpha=0.35,
                zorder=0
            )


def save_latex_table(plot_mean: pd.DataFrame, root: Path, flow_method: str):
    """导出 LaTeX 表（如不需要可注释掉 main() 中的调用）."""
    out_dir = root / "plot_absEPE"
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_file = out_dir / f"summary_abs_sigma{SIGMA:.2f}_{flow_method}.tex"
    plot_mean[["noise", "ΔNDMI%", "ΔEPE"]].to_latex(
        tex_file,
        index=False,
        float_format="%.2f",
        caption=f"Average ΔNDMI% and absolute ΔEPE for σ={SIGMA:.2f} ({flow_method}).",
        label=f"tab:mi_epe_abs_sigma{int(SIGMA*100)}_{flow_method}"
    )
    print(f"[LaTeX] {flow_method} → {tex_file}")


def main():
    # 读取两个方法数据
    data = {}
    for m in METHODS:
        plot_mean, root = load_delta_table(m)
        data[m] = {"df": plot_mean, "root": root}
        # 如不需要 LaTeX 输出，注释掉下一行
        save_latex_table(plot_mean, root, m)

    # 输出目录使用 flow_FB 的 root 下 plot_absEPE
    out_dir = data[METHODS[0]]["root"] / "plot_absEPE"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6, 8),
        constrained_layout=True
    )

    palette = sns.color_palette("deep", n_colors=3)
    group_order = ["affine", "gauss", "drift"]

    for idx, m in enumerate(METHODS):
        ax = axes[idx]
        df = data[m]["df"]

        # 各自坐标范围与刻度
        x_min, x_max = nice_limits(df["ΔNDMI%"].values)
        y_min, y_max = nice_limits(df["ΔEPE"].values)
        x_ticks = nice_ticks(x_min, x_max, n=5)
        y_ticks = nice_ticks(y_min, y_max, n=5)

        # 先画虚线参考线
        guide_lines(ax, x_min, x_max, y_min, y_max, idx)

        # 散点
        sns.scatterplot(
            data=df,
            x="ΔNDMI%",
            y="ΔEPE",
            hue="group",
            style="group",
            hue_order=group_order,
            style_order=group_order,
            s=70,
            edgecolor="black",
            linewidth=0.4,
            palette=palette,
            ax=ax,
            legend=(idx == 0)
        )

        # 0 参考线（更明显）
        ax.axvline(0, ls="--", lw=1.0, color="gray", alpha=0.65)
        ax.axhline(0, ls="--", lw=1.0, color="gray", alpha=0.65)

        # 应用范围与刻度
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        ax.grid(False)

        # 轴标签
        if idx == 0:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("ΔNDMI (%) vs baseline", fontsize=11)

        ax.set_ylabel("ΔEPE (pixels)", fontsize=11)
        ax.tick_params(axis="both", labelsize=9)

        # 图例
        if idx == 0:
            leg = ax.legend(
                title="Perturbation group",
                fontsize=8,
                title_fontsize=9,
                frameon=False,
                loc="upper right"
            )
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        # (a)/(b) 标签
        if idx == 0:
            ax.text(
                0.5, -0.08, "(a)",
                transform=ax.transAxes,
                fontsize=11,
                ha="center",
                va="top"
            )
        else:
            ax.text(
                0.5, -0.12, "(b)",
                transform=ax.transAxes,
                fontsize=11,
                ha="center",
                va="top"
            )

    png_file = out_dir / f"scatter_abs_sigma{SIGMA:.2f}_combined_FB_INR.png"
    plt.savefig(png_file, dpi=300)
    plt.close(fig)

    print("[FIG] Combined figure saved to:", png_file)


if __name__ == "__main__":
    main()
