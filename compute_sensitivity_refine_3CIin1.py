#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_sensitivity.py  –  Linear-error sensitivity & variance attribution
===========================================================================
功能概览
--------
* 自动解析 `<noise>` 名称 ⇒ 六维变量 (kappa, beta, sigma_I, delta_theta, delta_scale, dice_err)。
* 对每个链路（optical / inr / morph），相对 baseline 做差构建设计矩阵 X 与响应 y。
* 一阶线性化：最小二乘估计敏感度 g；报告 HC3 稳健 SE/CI 与病例簇自助 CI。
* 方差归因：按场景(all/intensity/gaussian/geometry) 计算总方差与各因子占比，并给出 bootstrap CI。
* 输出 CSV 与 Tornado 图（含 CI 的“all”场景森林图）。

修改日志
---------
2025-07-15
+ 自动 sigma_vec = X.std(ddof=1)。
+ 新增 "all" 场景并默认绘图；Tornado 设对数坐标并裁剪极小值。
+ HC3 使用 pinv 与 QR 计算 hat 值，兼容秩亏设计；全零列 SE 置 NaN。
"""

from __future__ import annotations
from pathlib import Path
import re, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 统一随机种子（用于 bootstrap）
rng = np.random.default_rng(42)

##############################################################################
# HC3 & Bootstrap
##############################################################################
def ols_hc3_se(X, y, beta, eps=1e-12, rcond=1e-10, return_details=False):
    """
    HC3 heteroskedasticity-robust SE for OLS under possible rank deficiency.

    - Hat diag via thin-QR (stable, no inverse).
    - Sandwich uses pinv(X'X) instead of inv(X'X).
    - Columns with ~zero norm are flagged and SE set to NaN.
    """
    n, p = X.shape
    # leverages via QR
    try:
        Q, R = np.linalg.qr(X, mode='reduced')
        h = np.sum(Q * Q, axis=1)
        rank = int(np.linalg.matrix_rank(R))
        sing_vals = np.linalg.svd(X, compute_uv=False)
    except Exception:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        h = np.sum(U * U, axis=1)
        rank = int(np.sum(s > rcond))
        sing_vals = s

    resid = y - X @ beta
    S = np.zeros((p, p), dtype=float)

    col_norm = np.linalg.norm(X, axis=0)
    zero_col = col_norm <= 1e-12

    for i in range(n):
        denom = max(1.0 - h[i], eps)
        ei = resid[i] / denom
        xi = X[i:i+1].T
        S += (ei * ei) * (xi @ xi.T)

    XtX = X.T @ X
    XtX_pinv = np.linalg.pinv(XtX, rcond=rcond)
    V = XtX_pinv @ S @ XtX_pinv
    se = np.sqrt(np.clip(np.diag(V), 0.0, np.inf))
    se[zero_col] = np.nan

    if return_details:
        return se, {"rank": rank, "singular_values": sing_vals, "hat_diag": h, "zero_col_mask": zero_col}
    return se


def cluster_bootstrap(X, y, case_ids, scen_sigma, B=2000):
    """Cluster bootstrap by case: resample unique cases with replacement."""
    cases = np.asarray(case_ids)
    unique_cases = np.unique(cases)
    if unique_cases.size == 0:
        raise ValueError("No case IDs available for bootstrap.")

    p = X.shape[1]
    G = []      # gradients (B, p)
    VF = {}     # scenario -> list of var_frac vectors
    VT = []     # total Var(S) under 'all'

    for _ in range(B):
        boot_cases = rng.choice(unique_cases, size=len(unique_cases), replace=True)
        idx = np.concatenate([np.where(cases == c)[0] for c in boot_cases])
        Xb, yb = X[idx], y[idx]
        try:
            gb = np.linalg.lstsq(Xb, yb, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        G.append(gb)
        for scen, sig in scen_sigma.items():
            varS = np.sum((gb * sig) ** 2)
            if varS < 1e-12:
                # 若该场景无有效方差，跳过
                continue
            vf = np.where(sig > 0, ((gb * sig) ** 2) / varS, 0.0)
            VF.setdefault(scen, []).append(vf)
            if scen == "all":
                VT.append(varS)

    if len(G) == 0:
        raise RuntimeError("Bootstrap produced no resamples (check data).")

    G = np.array(G)  # (B, p)

    out = {
        "g_median": np.median(G, axis=0),
        "g_lo": np.percentile(G, 2.5, axis=0),
        "g_hi": np.percentile(G, 97.5, axis=0),
    }

    vf_ci = {}
    for scen, mats in VF.items():
        M = np.array(mats)
        vf_ci[scen] = {
            "median": np.median(M, axis=0),
            "lo": np.percentile(M, 2.5, axis=0),
            "hi": np.percentile(M, 97.5, axis=0),
        }

    if len(VT) > 0:
        VT = np.array(VT)
        out["var_all_median"] = np.median(VT)
        out["var_all_lo"] = np.percentile(VT, 2.5)
        out["var_all_hi"] = np.percentile(VT, 97.5)

    return out, vf_ci


##############################################################################
# GLOBAL CONFIG – 修改路径即可运行
##############################################################################
FLOW_CSV   = Path("./MI_Jdet_flow_FB_curves/summary.csv")
INR_CSV    = Path("./MI_Jdet_flow_INR_curves/summary.csv")
MORPH_CSV  = Path("./MI_seg/morph_evaluation.csv")

OUT_DIR    = Path("./sensitivity_out")
SAVE_PLOTS = True

AFFINE_LEVELS = {  # affine_lvlk → (Δθ°, Δs)
    "1": (0.5,  0.02),
    "2": (1.0,  0.04),
    "3": (-1.0, -0.02),
}

# 变量顺序（必须与 X 的列顺序一致）
VARIABLES = ["kappa", "beta", "sigma_I", "delta_theta", "delta_scale", "dice_err"]

# 纵轴标签：用 MathText（无需安装 LaTeX）
VAR_LABEL = {
    "kappa":        r"$\kappa$",
    "beta":         r"$\beta$",
    "sigma_I":      r"$\sigma_I$",
    "delta_theta":  r"$\Delta\theta$",
    "delta_scale":  r"$\Delta s$",
    "dice_err":     r"$\mathrm{Dice\ err}$",   # 注意：mathtext 不支持 \text，改用 \mathrm
}

REGEX_RULES = [
    (re.compile(r"drift_k(\d{3})_b([+-]\d{2})"),
     lambda g: {"kappa": int(g[0]) / 100.0, "beta": int(g[1]) / 100.0}),
    (re.compile(r"gauss_sigma(\d{3})"),
     lambda g: {"sigma_I": int(g[0]) / 1000.0}),
    (re.compile(r"affine_lvl([123])"),
     lambda g: {"delta_theta": AFFINE_LEVELS[g[0]][0],
                "delta_scale": AFFINE_LEVELS[g[0]][1]}),
    (re.compile(r"baseline"),
     lambda g: {}),
]


##############################################################################
# Helpers
##############################################################################
def parse_noise(noise: str) -> dict[str, float]:
    vals = {v: 0.0 for v in VARIABLES}
    for rex, fn in REGEX_RULES:
        m = rex.fullmatch(noise)
        if m:
            vals.update(fn(m.groups()))
            return vals
    warnings.warn(f"[WARN] Noise '{noise}' not recognised; zeros assumed")
    return vals


def load_df(csv_path: Path, metric_col: str, chain: str) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[WARN] {csv_path} missing; skip {chain}", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    required = {"noise", "case", metric_col}
    if required - set(df.columns):
        raise ValueError(f"CSV {csv_path} must contain {required}")
    vars_df = df["noise"].apply(parse_noise).apply(pd.Series)
    df = pd.concat([df, vars_df], axis=1).rename(columns={metric_col: "metric"})
    return df


def add_dice_err(df: pd.DataFrame) -> pd.DataFrame:
    """dice_err = baseline_dice - current_dice（若无 dice 列则置 0）"""
    if "dice" not in df.columns:
        df["dice_err"] = 0.0
        return df
    base = df[df.noise == "baseline"].set_index("case")["dice"]
    def _row(r):
        b = base.get(r.case, np.nan)
        return (b - r.dice) if pd.notnull(b) else 0.0
    df["dice_err"] = df.apply(_row, axis=1)
    return df


def build_design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    基于“相对 baseline 差分”构建 (X, y)，并返回对齐的 case_id 向量。
    若某个 case 缺 baseline，则跳过该 case。
    """
    # 基于标记的 baseline 优先，其次用变量全零作为兜底
    base_mask = (df["noise"] == "baseline") | (df[VARIABLES].abs().sum(axis=1) < 1e-12)
    base = df[base_mask].drop_duplicates(subset=["case"]).set_index("case")

    rows = []
    keep_cases = []
    for _, r in df.iterrows():
        c = r.case
        if c not in base.index:
            continue
        br = base.loc[c]
        dx = r[VARIABLES].values - br[VARIABLES].values
        dy = r.metric - br.metric
        rows.append((*dx, dy))
        keep_cases.append(c)

    if len(rows) == 0:
        raise RuntimeError("No valid case with baseline found.")
    mat = np.array(rows, dtype=float)
    X = mat[:, :-1]
    y = mat[:, -1]
    return X, y, np.array(keep_cases)


def build_scen_sigma(variables, sigma_vec):
    """
    根据经验 σ（来自 X 的列标准差）构造各“场景”的 σ 向量。
    """
    p = len(variables)
    idx = {v: i for i, v in enumerate(variables)}
    def pick(*names):
        arr = np.zeros(p, dtype=float)
        for n in names:
            i = idx.get(n, None)
            if i is not None:
                arr[i] = sigma_vec[i]
        return arr

    scen = {
        "all":       sigma_vec.copy(),
        "intensity": pick("kappa", "beta"),
        "gaussian":  pick("sigma_I"),
        "geometry":  pick("delta_theta", "delta_scale"),
    }
    # 若 dice_err 有波动，也可选择保留独立场景
    if sigma_vec[idx["dice_err"]] > 0:
        scen["seg"] = pick("dice_err")
    return scen


##############################################################################
# MAIN
##############################################################################
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHAINS = {
    "optical": (FLOW_CSV,  "mean"),
    "inr":     (INR_CSV,   "mean"),
    "morph":   (MORPH_CSV, "delta_area_mi"),
}

# 收集各方法 "all" 场景的 CI，便于最后合成图
vf_ci_all = {}   # {method: {"median":..., "lo":..., "hi":...}}

for cname, (csv_path, metric_col) in CHAINS.items():
    df = load_df(csv_path, metric_col, cname)
    if df.empty:
        continue

    if cname == "morph":
        df = add_dice_err(df)
    else:
        df["dice_err"] = 0.0

    # —— 构建设计矩阵（对齐 case 向量）——
    try:
        X, y, cases_vec = build_design(df)
    except RuntimeError as e:
        print(f"[WARN] {cname}: {e}; skip.", file=sys.stderr)
        continue

    # —— 敏感度：最小二乘（允许秩亏时取最小范数解）——
    g = np.linalg.lstsq(X, y, rcond=None)[0]   # shape (6,)

    # —— 经验 σ ——
    sigma_vec = X.std(axis=0, ddof=1)          # shape (6,)
    print(f"{cname}: empirical σ →", dict(zip(VARIABLES, sigma_vec)))

    # —— 场景定义（务必在 bootstrap 前构造）——
    scen_sigma = build_scen_sigma(VARIABLES, sigma_vec)

    # —— PARAMETRIC: HC3 robust SE / CI for gradients ——
    se, det = ols_hc3_se(X, y, g, return_details=True)
    if det["rank"] < X.shape[1]:
        print(f"[INFO] {cname}: rank-deficient design (rank={det['rank']}/{X.shape[1]}).", file=sys.stderr)
    z = 1.96
    g_ci_lo = g - z * se
    g_ci_hi = g + z * se
    pd.DataFrame({
        "variable": VARIABLES,
        "g_point": g,
        "g_se_hc3": se,
        "g_ci_lo_hc3": g_ci_lo,
        "g_ci_hi_hc3": g_ci_hi,
    }).to_csv(OUT_DIR / f"{cname}_gradients_parametric.csv", index=False)

    # —— NONPARAMETRIC: cluster bootstrap (by case) ——
    boot_out, vf_ci = cluster_bootstrap(X, y, cases_vec, scen_sigma, B=2000)

    # 记录本方法的 "all" 场景 CI（若存在）
    if "all" in vf_ci:
        vf_ci_all[cname] = {
            "median": np.asarray(vf_ci["all"]["median"], float),
            "lo": np.asarray(vf_ci["all"]["lo"], float),
            "hi": np.asarray(vf_ci["all"]["hi"], float),
        }

    # 保存梯度的 bootstrap 区间
    pd.DataFrame({
        "variable": VARIABLES,
        "g_bt_median": boot_out["g_median"],
        "g_bt_ci_lo": boot_out["g_lo"],
        "g_bt_ci_hi": boot_out["g_hi"],
    }).to_csv(OUT_DIR / f"{cname}_gradients_bootstrap.csv", index=False)

    # 保存各场景方差贡献比例 CI
    rows = []
    for scen, ci in vf_ci.items():
        for v, med, lo, hi in zip(VARIABLES, ci["median"], ci["lo"], ci["hi"]):
            rows.append({
                "scenario": scen, "variable": v,
                "var_frac_median": med,
                "var_frac_ci_lo": lo,
                "var_frac_ci_hi": hi
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / f"{cname}_var_contrib_bootstrap.csv", index=False)

    # 保存总方差 CI（all 场景）
    if "var_all_median" in boot_out:
        pd.DataFrame([{
            "var_all_median": boot_out["var_all_median"],
            "var_all_ci_lo": boot_out["var_all_lo"],
            "var_all_ci_hi": boot_out["var_all_hi"],
        }]).to_csv(OUT_DIR / f"{cname}_var_total_bootstrap.csv", index=False)

    # —— 保存梯度点估计（便于核对）——
    pd.DataFrame({"variable": VARIABLES, "sensitivity": g}).to_csv(
        OUT_DIR / f"{cname}_gradients.csv", index=False
    )

    # —— 方差占比点估计（非 CI 版）——
    var_rows = []
    for scen, sig in scen_sigma.items():
        varS = np.sum((g * sig) ** 2)
        if varS < 1e-12:
            continue
        for v, gi, sgi in zip(VARIABLES, g, sig):
            frac = (gi * sgi) ** 2 / varS if sgi else 0.0
            var_rows.append({"scenario": scen, "variable": v, "var_frac": frac})
    pd.DataFrame(var_rows).to_csv(OUT_DIR / f"{cname}_var_contrib.csv", index=False)

    # —— 画带置信区间的 Tornado（仅 all 场景的 CI 森林图）——
    if SAVE_PLOTS and ("all" in vf_ci):
        ci = vf_ci["all"]
        order = np.argsort(ci["median"])
        names = np.array(VARIABLES)[order]
        med = ci["median"][order]
        lo  = ci["lo"][order]
        hi  = ci["hi"][order]
        plt.figure(figsize=(6.2, 3.8))
        y_idx = np.arange(len(names))
        plt.barh(y_idx, np.clip(med, 1e-6, 1), edgecolor="k")
        for i in range(len(names)):
            plt.plot([max(lo[i], 1e-6), max(hi[i], 1e-6)], [y_idx[i], y_idx[i]], linewidth=2)
        plt.xscale("log")
        plt.yticks(y_idx, names)
        plt.xlabel("Fraction of Var(S) (median with 95% CI)")
        plt.title(f"{cname.capitalize()} – all (bootstrap)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{cname}_all_tornado_CI.png", dpi=150)
        plt.close()

print(f"[DONE] Sensitivity results saved to {OUT_DIR.resolve()}")

# ============== 合成 Tornado 图（2×2：上2下1；右下空；无标题） ==============
try:
    order_methods = ["optical", "morph", "inr"]
    order_methods = [m for m in order_methods if m in vf_ci_all]
    if len(order_methods) >= 1:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import numpy as np

        # 颜色：柱 C0（蓝），CI 线按色盘；若蓝则换下一色
        color_cycle = plt.rcParams.get("axes.prop_cycle", None)
        cycle_colors = (color_cycle.by_key()["color"] if color_cycle
                        else ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"])
        bar_color = cycle_colors[0]  # C0 蓝

        # 全局最小正值（含 median/lo/hi），左界取其一半，保证最小 dice_err 也可见
        pos_vals = []
        for m in order_methods:
            ci = vf_ci_all[m]
            for arr in (ci["median"], ci["lo"], ci["hi"]):
                a = np.asarray(arr, float)
                pos = a[a > 0]
                if pos.size:
                    pos_vals.append(pos.min())
        global_min = min(pos_vals) if pos_vals else 1e-6
        xmin = max(global_min * 0.5, 1e-12)
        xmax = 1.0

        if len(order_methods) == 3:
            # 2×2：上面 optical/morph，下面 inr 在左；右下空白
            fig = plt.figure(figsize=(12.6, 8.2))
            gs  = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.26)
            ax_map = {
                "optical": fig.add_subplot(gs[0, 0]),
                "morph":   fig.add_subplot(gs[0, 1]),
                "inr":     fig.add_subplot(gs[1, 0]),
            }
            ax_empty = fig.add_subplot(gs[1, 1]); ax_empty.axis("off")
            panel_tags = {"optical":"(a)", "morph":"(b)", "inr":"(c)"}
        elif len(order_methods) == 2:
            fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=False)
            ax_map = {order_methods[0]: axes[0], order_methods[1]: axes[1]}
            panel_tags = {order_methods[0]:"(a)", order_methods[1]:"(b)"}
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.8))
            ax_map = {order_methods[0]: ax}
            panel_tags = {order_methods[0]:"(a)"}

        for m, ax in ax_map.items():
            if m not in vf_ci_all:  # 跳过空白轴
                continue
            ci  = vf_ci_all[m]
            med = np.asarray(ci["median"], float)
            lo  = np.asarray(ci["lo"],     float)
            hi  = np.asarray(ci["hi"],     float)

            # 每个面板按自身中位占比升序排序；各自 y 轴标签（数学符号）
            order = np.argsort(med)
            med_s = np.clip(med[order], xmin, xmax)
            lo_s  = np.clip(lo[order],  xmin, xmax)
            hi_s  = np.clip(hi[order],  xmin, xmax)
            names = [VAR_LABEL[v] for v in np.array(VARIABLES)[order]]
            y_idx = np.arange(len(names))

            # 柱：C0 蓝，细黑边
            ax.barh(y_idx, med_s, color=bar_color, edgecolor="k", linewidth=0.6, alpha=0.92)

            # CI 线：彩色；若恰是蓝则换下一色
            for i in range(len(y_idx)):
                ci_col = cycle_colors[i % len(cycle_colors)]
                if ci_col.lower() in ("c0", "#1f77b4") or ci_col.lower() == bar_color.lower():
                    ci_col = cycle_colors[(i + 1) % len(cycle_colors)]
                ax.hlines(y_idx[i], lo_s[i], hi_s[i], colors=ci_col, linewidth=2)

            ax.set_xscale("log")
            ax.set_xlim(left=xmin, right=xmax)
            ax.set_ylim(-0.5, len(y_idx) - 0.5)

            ax.set_yticks(y_idx)
            ax.set_yticklabels(names, fontsize=11)

            ax.set_xlabel("Fraction of Var(S) (median with 95% CI)", fontsize=11, labelpad=4)
            ax.tick_params(axis="x", pad=2)

            # (a)(b)(c) 面板标注：字号 14，位置靠近图，不裁剪
            ax.set_title("")
            ax.text(0.5, -0.18, panel_tags[m], transform=ax.transAxes,
                    ha="center", va="top", fontsize=14, fontweight="bold", clip_on=False)

            ax.grid(True, axis="x", linestyle="--", alpha=0.35)
            ax.grid(False, axis="y")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

        plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.14)
        out_path = OUT_DIR / "tornado_all_composite.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[SAVED] Composite tornado → {out_path}")
except Exception as e:
    print("[WARN] Composite tornado skipped due to:", repr(e))



