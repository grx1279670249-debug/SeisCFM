# -*- coding: utf-8 -*-
"""
PGA 统计分布检验与可视化工具箱（适用于 numpy 数据）
==================================================
功能：
- 逐分量（X/Y/Z 或 E/N/Z）与全量（3 分量合并）的分布检验：
  * KS 双样本检验（ks_2samp）
  * Anderson-Darling k-sample（anderson_ksamp）
  * 一维 Wasserstein 距离（EMD）
  * Cliff's δ（基于 Mann-Whitney U 计算）
  * 均值/标准差/偏度/峰度、分位数差（包括尾部）
- 可视化：直方图（FD 规则自动分箱）+ ECDF、QQ 图
- 结果导出为 CSV；图像保存为 PNG。

使用方式：
1) 在你的代码里 `import pga_stats_toolkit as pga`；
2) 调用 `pga.run_all(pga_pred, pga_true, outdir="pga_eval")`；
3) 在 outdir 下查看 `metrics_summary.csv`、各类图像。

注意：
- 输入数组形状应为 (N, 3, 1) 或 (N, 3)，单位需一致（例如均为 g）。
- 自动去除 NaN/Inf；支持可选的绝对值裁剪（例如物理上不合理的异常值）。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List, Optional
from scipy import stats


# ----------------------------- 工具函数 -----------------------------

def _ensure_shape(x: np.ndarray) -> np.ndarray:
    """将 (N, 3, 1) 或 (N, 3) 统一到 (N, 3)。"""
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    assert x.ndim == 2 and x.shape[1] == 3, f"期望形状 (N,3) 或 (N,3,1)，但得到 {x.shape}"
    return x


def _clean(x: np.ndarray) -> np.ndarray:
    """去除 NaN/Inf。"""
    x = np.asarray(x).astype(float)
    mask = np.isfinite(x)
    return x[mask]


def _fd_bins(x: np.ndarray, min_bins: int = 20, max_bins: int = 100) -> int:
    """Freedman–Diaconis 规则自动选择分箱数，限制在区间内。"""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return min_bins
    iqr = np.subtract(*np.nanpercentile(x, [75, 25]))
    if iqr == 0:
        return min_bins
    bin_width = 2 * iqr * (n ** (-1 / 3))
    if bin_width <= 0:
        return min_bins
    bins = int(np.ceil((np.nanmax(x) - np.nanmin(x)) / bin_width))
    return max(min_bins, min(max_bins, bins if bins > 0 else min_bins))


def _ecdf(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """经验分布函数（ECDF）。返回有序样本及其累计概率。"""
    y = np.sort(y)
    n = y.size
    p = np.arange(1, n + 1) / n
    return y, p


def _cliffs_delta_from_mwu(x: np.ndarray, y: np.ndarray) -> float:
    """
    用 Mann-Whitney U 近似计算 Cliff's delta：
    δ = 2U / (n*m) - 1
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n, m = x.size, y.size
    if n == 0 or m == 0:
        return np.nan
    # 使用双侧检验，U 取较小者；为得到 δ 需要 U 的“较大者”，因此取 U1 = n*m - U_small
    U_small, _ = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
    U_large = n * m - U_small
    delta = 2.0 * U_large / (n * m) - 1.0
    return float(delta)


def _quantile_grid_diff(x: np.ndarray, y: np.ndarray,
                        qs: List[float] = (1, 5, 10, 25, 50, 75, 90, 95, 99)) -> Dict[str, float]:
    """按给定百分位网格比较两分布的分位数差异。返回 L1/L2/最大绝对差等摘要。"""
    xq = np.nanpercentile(x, qs)
    yq = np.nanpercentile(y, qs)
    diff = xq - yq
    out = {
        "q_diff_L1": float(np.abs(diff).mean()),
        "q_diff_L2": float(np.sqrt(np.mean(diff ** 2))),
        "q_diff_Linf": float(np.max(np.abs(diff))),
        "q50_diff": float(diff[qs.index(50)]),
        "q95_diff": float(diff[qs.index(95)]),
        "q99_diff": float(diff[qs.index(99)]),
    }
    return out


def _basic_stats(x: np.ndarray) -> Dict[str, float]:
    """基础统计量。"""
    return {
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x, ddof=1)) if x.size > 1 else np.nan,
        "skew": float(stats.skew(x, bias=False, nan_policy="omit")),
        "kurtosis": float(stats.kurtosis(x, fisher=True, bias=False, nan_policy="omit")),
        "p1": float(np.nanpercentile(x, 1)),
        "p5": float(np.nanpercentile(x, 5)),
        "p50": float(np.nanpercentile(x, 50)),
        "p95": float(np.nanpercentile(x, 95)),
        "p99": float(np.nanpercentile(x, 99)),
    }


# ----------------------------- 指标计算核心 -----------------------------
def compute_metrics_1d(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """对一维向量（单分量或已合并）计算分布检验与效应量。"""
    pred = _clean(np.ravel(pred))
    true = _clean(np.ravel(true))

    res: Dict[str, float] = {
        "n_pred": float(pred.size),
        "n_true": float(true.size),
    }
    # 基础统计
    res.update({f"pred_{k}": v for k, v in _basic_stats(pred).items()})
    res.update({f"true_{k}": v for k, v in _basic_stats(true).items()})

    # KS
    ks = stats.ks_2samp(pred, true, alternative="two-sided", method="auto")
    res["ks_stat"], res["ks_p"] = float(ks.statistic), float(ks.pvalue)

    # AD (k-sample)
    try:
        ad = stats.anderson_ksamp([pred, true], method=stats.PermutationMethod())
        # SciPy 返回 statistic 与 significance_level（不是严格意义的 p-value；越小越显著）
        res["ad_stat"] = float(ad.statistic)
        res["ad_significance_level"] = float(ad.significance_level)
    except Exception:
        res["ad_stat"] = np.nan
        res["ad_significance_level"] = np.nan

    # Wasserstein（EMD）
    try:
        res["wasserstein"] = float(stats.wasserstein_distance(pred, true))
    except Exception:
        res["wasserstein"] = np.nan

    # Cliff's delta
    try:
        res["cliffs_delta"] = _cliffs_delta_from_mwu(pred, true)
    except Exception:
        res["cliffs_delta"] = np.nan

    # 分位数差
    res.update(_quantile_grid_diff(pred, true))

    return res

# ----------------------------- 主流程 -----------------------------

def run_all(pga_pred: np.ndarray,
            pga_true: np.ndarray,
            outdir: str = "pga_eval",
            clip_abs: Optional[float] = None) -> pd.DataFrame:
    """
    运行逐分量与合并分布检验与可视化。

    参数
    ----
    pga_pred, pga_true: numpy 数组，形状 (N,3,1) 或 (N,3)，单位一致
    outdir: 输出目录
    clip_abs: 可选，对 |PGA| 进行裁剪上限（例如 clip_abs=2.0 表示 |PGA|>2.0 的值将被截断为 2.0），
              用于少量异常值的温和控制；默认不裁剪。

    返回
    ----
    metrics_df: 汇总指标的 DataFrame，并保存为 outdir/metrics_summary.csv
    """
    os.makedirs(outdir, exist_ok=True)

    pred = _ensure_shape(pga_pred)
    true = _ensure_shape(pga_true)

    if clip_abs is not None and clip_abs > 0:
        pred = np.clip(pred, -clip_abs, clip_abs)
        true = np.clip(true, -clip_abs, clip_abs)

    metrics_rows = []
    names = ["comp1", "comp2", "comp3", "all"]

    # 逐分量
    for i in range(3):
        m = compute_metrics_1d(pred[:, i], true[:, i])
        m["component"] = names[i]
        metrics_rows.append(m)

    # 合并（3 分量拼接）
    m_all = compute_metrics_1d(pred.reshape(-1), true.reshape(-1))
    m_all["component"] = "all"
    metrics_rows.append(m_all)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[["component"] + [c for c in metrics_df.columns if c != "component"]]
    metrics_df.to_csv(os.path.join(outdir, "metrics_summary.csv"), index=False)
    return metrics_df


# 可作为脚本直接运行的演示（会在本地生成 pga_eval_demo/）
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 2000
    true_demo = rng.lognormal(mean=-2.0, sigma=0.6, size=(N, 3))
    pred_demo = rng.lognormal(mean=-2.05, sigma=0.62, size=(N, 3))
    run_all(pred_demo, true_demo, outdir="pga_eval_demo", clip_abs=None)
