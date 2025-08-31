#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal IC and bin analysis with a simple HTML dashboard.

Supports two analysis modes:
- cross: cross-sectional per-date IC (Spearman) and per-date rank-binning
- global: fallback when cross-sections are sparse. Uses overall IC and global bins;
          IC time series is a rolling correlation of daily-mean pred vs gt.

Usage:
  python scripts/signal_dashboard.py \
    --csv /path/to/results.csv \
    --out ./signal_analysis_out \
    --bins 20 \
    --mode auto \
    --roll 60 \
    --date 2025-08-20

Expected columns:
- instrument_name, test_date
- pred_return_{k}_days, gt_return_{k}_days for horizons k
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Matplotlib style fallback
try:
    plt.style.use("seaborn-v0_8")
except Exception:
    try:
        plt.style.use("seaborn")
    except Exception:
        try:
            plt.style.use("ggplot")
        except Exception:
            pass


@dataclass
class ICSummary:
    horizon: int
    n_days: int
    mean: float
    std: float
    ir: float
    t_stat_norm: float
    p_value_norm: float
    pos_ratio: float


@dataclass
class LSSummary:
    horizon: int
    n_days: int
    mean: float
    std: float
    ir: float
    t_stat_norm: float
    p_value_norm: float
    pos_ratio: float
    cum_return: float


# ----------------------------- utils -----------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _spearman_no_scipy(x: pd.Series, y: pd.Series) -> float:
    s = pd.concat([x, y], axis=1).dropna()
    if s.shape[0] < 2:
        return np.nan
    xr = s.iloc[:, 0].rank(method="average")
    yr = s.iloc[:, 1].rank(method="average")
    return float(xr.corr(yr))


def _norm_p_value_from_t(t_stat: float) -> float:
    from math import erf, sqrt

    if t_stat is None or np.isnan(t_stat):
        return np.nan
    z = abs(t_stat)
    phi = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    return float(2.0 * (1.0 - phi))


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    try:
        return f"{v:.{digits}f}"
    except Exception:
        return str(v)


# Rank-normalize a series to [-1, 1] using Spearman-style ranks.
def spearman_normalize(s: pd.Series) -> pd.Series:
    r = s.rank(method="average")  # 1..n for non-NaNs, NaN stays NaN
    n = int(r.notna().sum())
    if n <= 1:
        out = pd.Series(np.nan, index=s.index)
        out.loc[s.notna()] = 0.0
        return out
    x = (r - 1.0) / (n - 1.0)  # map to [0,1]
    x = x * 2.0 - 1.0  # map to [-1,1]
    return x


# -------------------------- detection ----------------------------


def detect_horizons(columns: List[str]) -> List[int]:
    patt = re.compile(r"^pred_return_(\d+)_days$")
    horizons: List[int] = []
    for c in columns:
        m = patt.match(c)
        if not m:
            continue
        k = int(m.group(1))
        if f"gt_return_{k}_days" in columns:
            horizons.append(k)
    return sorted(set(horizons))


def cross_section_counts(
    df: pd.DataFrame, date_col: str, instrument_col: str
) -> pd.DataFrame:
    c = (
        df.groupby(date_col)[instrument_col]
        .nunique()
        .rename("n_instruments")
        .reset_index()
    )
    return c


# -------------------------- IC metrics ---------------------------


def compute_ic_daily(
    df: pd.DataFrame, pred_col: str, gt_col: str, date_col: str
) -> pd.DataFrame:
    def _ic_for_date(g: pd.DataFrame) -> float:
        return _spearman_no_scipy(g[pred_col], g[gt_col])

    ic = df.groupby(date_col, sort=True).apply(_ic_for_date).rename("ic").reset_index()
    return ic


def summarize_ic(ic_df: pd.DataFrame, horizon: int) -> ICSummary:
    vals = ic_df["ic"].dropna().values
    n = len(vals)
    if n == 0:
        return ICSummary(horizon, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else np.nan
    ir = mean / std if (std and not np.isnan(std) and std != 0.0) else np.nan
    t_stat = (
        (mean / std * np.sqrt(n))
        if (std and not np.isnan(std) and std != 0.0)
        else np.nan
    )
    p_val = _norm_p_value_from_t(t_stat)
    pos_ratio = float(np.mean(vals > 0.0)) if n > 0 else np.nan
    return ICSummary(horizon, n, mean, std, ir, t_stat, p_val, pos_ratio)


# --------------------------- Bin logic ---------------------------


def bin_by_rank(pred: pd.Series, bins: int) -> pd.Series:
    s = pred.rank(method="first")
    n = s.shape[0]
    if n == 0:
        return pd.Series([], dtype=int)
    bins_eff = int(max(1, min(bins, n)))
    b = np.floor((s - 1) * bins_eff / n).astype(int)
    b = b.clip(lower=0, upper=bins_eff - 1)
    return pd.Series(b, index=pred.index, dtype=int)


def compute_bins_daily(
    df: pd.DataFrame,
    pred_col: str,
    gt_col: str,
    date_col: str,
    bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns in order:
      - bins_daily: index=date, columns=bin_1..bin_B, mean alpha per bin per day
        where alpha_t,i = gt_t,i - mean_i(gt_t,·) (cross-sectional mean per date)
      - bins_mean: columns [avg_alpha, std_alpha, count_days]
      - counts_daily: index=date, columns=bin_1..bin_B, counts per bin per day
      - ls_daily: columns [date, spread]
    """
    daily_bins: Dict[pd.Timestamp, np.ndarray] = {}
    daily_counts: Dict[pd.Timestamp, np.ndarray] = {}
    ls_records: List[Tuple[pd.Timestamp, float]] = []

    for dt, g in df.groupby(date_col, sort=True):
        gg = g[[pred_col, gt_col]].dropna()
        dt = pd.to_datetime(dt)
        if gg.empty:
            daily_bins[dt] = np.full(bins, np.nan, dtype=float)
            daily_counts[dt] = np.zeros(bins, dtype=int)
            ls_records.append((dt, np.nan))
            continue
        codes = bin_by_rank(gg[pred_col], bins=bins)
        gg = gg.assign(_bin=codes.values)
        # Cross-sectional market proxy: per-date mean of gt over available instruments
        m_t = float(gg[gt_col].mean())
        bin_means = gg.groupby("_bin")[gt_col].mean()
        bin_counts = gg.groupby("_bin")[gt_col].size()
        arr = np.full(bins, np.nan, dtype=float)
        cnt = np.zeros(bins, dtype=int)
        for b_idx, val in bin_means.items():
            # alpha per bin = mean(gt) - mean_market_t
            arr[int(b_idx)] = float(val) - m_t
        for b_idx, c in bin_counts.items():
            cnt[int(b_idx)] = int(c)
        daily_bins[dt] = arr
        daily_counts[dt] = cnt
        # Use first/last non-NaN bin to avoid empty extremes when bins > n
        nz_idx = np.where(~np.isnan(arr))[0]
        if nz_idx.size >= 2:
            bot = arr[nz_idx[0]]
            top = arr[nz_idx[-1]]
            spread = float(top - bot)
        else:
            spread = np.nan
        ls_records.append((dt, spread))

    dates = sorted(daily_bins.keys())
    bins_cols = [f"bin_{i+1}" for i in range(bins)]
    bins_daily = pd.DataFrame(
        [daily_bins[d] for d in dates], index=dates, columns=bins_cols
    )
    counts_daily = pd.DataFrame(
        [daily_counts[d] for d in dates], index=dates, columns=bins_cols
    )
    bins_mean = bins_daily.mean(axis=0, skipna=True).to_frame(name="avg_alpha")
    bins_mean["std_alpha"] = bins_daily.std(axis=0, skipna=True)
    bins_mean["count_days"] = bins_daily.notna().sum(axis=0)

    ls_daily = pd.DataFrame(ls_records, columns=["date", "spread"]).sort_values("date")
    return bins_daily, bins_mean, counts_daily, ls_daily


def compute_bins_global(
    df: pd.DataFrame,
    pred_col: str,
    gt_col: str,
    date_col: str,
    bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Global rank-binning over all rows, then aggregate per date."""
    d = df[[date_col, pred_col, gt_col]].dropna().copy()
    if d.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, pd.DataFrame({"date": [], "spread": []})

    codes = bin_by_rank(d[pred_col], bins=bins)
    d["_bin"] = codes.values

    # Per-date, per-bin average gt
    bins_daily = (
        d.groupby([date_col, "_bin"], sort=True)[gt_col]
        .mean()
        .unstack("_bin")
        .rename(columns=lambda i: f"bin_{i+1}")
        .sort_index()
    )
    # Convert to per-date per-bin alpha by subtracting per-date cross-sectional mean
    m_series = d.groupby(date_col)[gt_col].mean()
    bins_daily = bins_daily.sub(m_series, axis=0)
    # Counts per date per bin
    counts_daily = (
        d.groupby([date_col, "_bin"], sort=True)[gt_col]
        .size()
        .unstack("_bin")
        .rename(columns=lambda i: f"bin_{i+1}")
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    # Mean across dates
    bins_mean = pd.DataFrame(
        {
            "avg_alpha": bins_daily.mean(axis=0, skipna=True),
            "std_alpha": bins_daily.std(axis=0, skipna=True),
            "count_days": bins_daily.notna().sum(axis=0),
        }
    )

    # Long-short per date
    # Compute spread using first/last non-NaN per row
    if not bins_daily.empty:

        def _row_spread(row: pd.Series) -> float:
            vals = row.values.astype(float)
            nz = np.where(~np.isnan(vals))[0]
            if nz.size >= 2:
                return float(vals[nz[-1]] - vals[nz[0]])
            return np.nan

        ls_daily = pd.DataFrame(
            {
                "date": bins_daily.index,
                "spread": bins_daily.apply(_row_spread, axis=1).values,
            }
        )
    else:
        ls_daily = pd.DataFrame({"date": bins_daily.index, "spread": np.nan})

    return bins_daily, bins_mean, counts_daily, ls_daily


# -------------------------- LS summary ---------------------------


def summarize_ls(ls_daily: pd.DataFrame, horizon: int) -> LSSummary:
    vals = (
        ls_daily["spread"].dropna().values
        if "spread" in ls_daily.columns
        else np.array([])
    )
    n = len(vals)
    if n == 0:
        return LSSummary(
            horizon, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        )
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else np.nan
    ir = mean / std if (std and not np.isnan(std) and std != 0.0) else np.nan
    t_stat = (
        (mean / std * np.sqrt(n))
        if (std and not np.isnan(std) and std != 0.0)
        else np.nan
    )
    p_val = _norm_p_value_from_t(t_stat)
    pos_ratio = float(np.mean(vals > 0.0)) if n > 0 else np.nan
    cum_return = float(np.nansum(vals))
    return LSSummary(horizon, n, mean, std, ir, t_stat, p_val, pos_ratio, cum_return)


# ---------------------------- plots ------------------------------


def plot_ic_timeseries(ic_df: pd.DataFrame, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = ic_df.iloc[:, 0]
    y = ic_df["ic"].values
    if len(ic_df) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    elif len(ic_df) == 1:
        ax.scatter(x, y, color="#1f77b4")
    else:
        ax.plot(x, y, label="IC", color="#1f77b4", marker="o", linewidth=1)
    ax.axhline(0, color="grey", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("IC (Spearman)")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ic_hist(ic_df: pd.DataFrame, out_path: str, title: str) -> None:
    vals = ic_df["ic"].dropna().values
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if len(vals) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    elif len(vals) < 3:
        # Small-n: show informative markers and text instead of misleading histogram
        for v in vals:
            ax.axvline(float(v), color="#1f77b4", linestyle="-", linewidth=6, alpha=0.6)
        ax.text(
            0.02,
            0.95,
            f"n={len(vals)} (histogram not informative)",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="#444",
        )
    else:
        # Adaptive binning: Freedman–Diaconis with fallbacks
        q75, q25 = np.percentile(vals, [75, 25])
        iqr = float(q75 - q25)
        n = len(vals)
        if iqr > 0:
            width = 2.0 * iqr * (n ** (-1.0 / 3.0))
            span = float(np.max(vals) - np.min(vals))
            bins_fd = int(np.ceil(span / width)) if width > 0 else 10
        else:
            bins_fd = 10
        bins_sqrt = max(5, int(np.sqrt(n) * 3))
        bins = int(np.clip(bins_fd, 5, 50)) if n >= 3 else bins_sqrt
        ax.hist(vals, bins=bins, color="#1f77b4", alpha=0.8, edgecolor="white")

        # Rug marks along the x-axis
        y0, y1 = ax.get_ylim()
        y_rug = y0 + 0.03 * (y1 - y0)
        ax.scatter(
            vals,
            np.full_like(vals, y_rug),
            marker="|",
            s=120,
            color="#1f77b4",
            alpha=0.7,
        )

        # Stats and reference lines
        mean_ic = float(np.mean(vals))
        std_ic = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        pos_ratio = float(np.mean(vals > 0.0))
        ax.axvline(0.0, color="grey", linestyle=":", linewidth=1)
        ax.axvline(mean_ic, color="#d62728", linestyle="-", linewidth=1.5, label="mean")
        txt = f"n={n}\nmean={fmt(mean_ic,3)}\nstd={fmt(std_ic,3)}\n>0={fmt(pos_ratio*100,1)}%"
        ax.text(
            0.98,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd"),
        )
        ax.legend(loc="upper left", frameon=False)

    ax.set_title(title)
    ax.set_xlabel("IC")
    ax.set_ylabel("Frequency")
    ax.set_xlim(-1.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rank_scatter(
    df: pd.DataFrame, pred_col: str, gt_col: str, out_path: str, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    d = df[[pred_col, gt_col]].dropna()
    if d.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        xr = spearman_normalize(d[pred_col])
        yr = spearman_normalize(d[gt_col])
        # xr = d[pred_col]
        # yr = d[gt_col]
        ax.scatter(
            xr.values, yr.values, s=14, alpha=0.6, color="#1f77b4", edgecolors="none"
        )
        # y=x reference
        ax.plot([-1, 1], [-1, 1], linestyle="--", color="grey", linewidth=1)
        rho = _spearman_no_scipy(d[pred_col], d[gt_col])
        ax.set_title(f"{title}\nSpearman rho={fmt(rho,3)}, n={len(d)}")
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_xlabel("Pred (rank-normalized)")
        ax.set_ylabel("GT (rank-normalized)")
        ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bins_bar(bins_mean: pd.DataFrame, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ylabel = ""
    if bins_mean is None or bins_mean.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        key = (
            "avg_alpha"
            if "avg_alpha" in bins_mean.columns
            else ("avg_gt" if "avg_gt" in bins_mean.columns else None)
        )
        if key is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            y = bins_mean[key].values.astype(float)
            mask = ~np.isnan(y)
            y = y[mask]
            if y.size == 0:
                ax.text(
                    0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
                )
            else:
                x = np.arange(1, len(y) + 1)
                ax.bar(x, y, color="#2ca02c", alpha=0.9)
            ylabel = "Avg Alpha Return" if key == "avg_alpha" else "Avg GT Return"
    ax.set_title(title)
    ax.set_xlabel("Bin (1=Short, B=Long)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bin_counts(counts_daily: pd.DataFrame, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if counts_daily is None or counts_daily.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        counts_mean = counts_daily.mean(axis=0)
        y = counts_mean.values.astype(float)
        x = np.arange(1, len(y) + 1)
        bars = ax.bar(x, y, color="#6a5acd", alpha=0.9)
        for rect, val in zip(bars, y):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height(),
                f"{int(round(val))}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_title(title)
    ax.set_xlabel("Bin")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lift_curve(df: pd.DataFrame, pred_col: str, gt_col: str, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    d = df[[pred_col, gt_col]].dropna()
    if d.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    elif len(d) == 1:
        v = float(d.iloc[0][gt_col])
        ax.bar([1], [v], color="#ff7f0e", alpha=0.9)
        ax.set_xticks([1])
        ax.set_xticklabels(["Top 100%"])
        ax.text(1, v, f" {v:.6f}", va="bottom" if v >= 0 else "top")
    else:
        d = d.sort_values(pred_col, ascending=False)
        y = d[gt_col].values.astype(float)
        n = len(y)
        overall = float(np.mean(y))
        cum_mean = np.cumsum(y) / np.arange(1, n + 1)
        lift = cum_mean - overall
        x = np.arange(1, n + 1) / float(n)
        ax.plot(x, lift, color="#ff7f0e", linewidth=2, label="Lift (cum mean - overall)")
        ax.axhline(0.0, color="grey", linestyle=":", linewidth=1)
        for k in (0.1, 0.2):
            xi = max(1, int(round(k * n)))
            xv = xi / float(n)
            ax.axvline(xv, color="#ddd", linestyle=":", linewidth=1)
            ax.text(xv, lift[xi - 1], f" {int(k*100)}%: {fmt(lift[xi-1],3)}", va="bottom", fontsize=9)
        ax.legend(loc="best", frameon=False)
        ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Coverage (top-k%)")
    ax.set_ylabel("Lift vs overall mean")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_long_short_curve(ls_daily: pd.DataFrame, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if ls_daily is None or ls_daily.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        series = ls_daily.set_index("date")["spread"].dropna().sort_index()
        if len(series) <= 1:
            # Single-date: replace curve with a single bar showing LS spread
            if len(series) == 1:
                val = float(series.iloc[0])
                ax.bar([1], [val], color="#d62728", alpha=0.9)
                ax.set_xticks([1])
                ax.set_xticklabels(["LS spread"])
                ax.set_ylabel("Spread (Top-Bottom)")
                # annotate value
                ax.text(1, val, f" {val:.6f}", va="bottom" if val >= 0 else "top")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            s = series.fillna(0.0).cumsum()
            ax.plot(
                s.index,
                s.values,
                color="#d62728",
                label="Long-Short Cumulative",
                marker="o",
                linewidth=1,
            )
    ax.axhline(0, color="grey", linewidth=1)
    ax.set_title(title)
    # X/Y labels depend on mode drawn above
    if (
        ls_daily is not None
        and not ls_daily.empty
        and len(ls_daily.set_index("date")["spread"].dropna()) <= 1
    ):
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return (Top-Bottom)")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -------------------------- dashboard ---------------------------


def build_dashboard(
    out_root: str,
    horizons: List[int],
    ic_summaries: Dict[int, ICSummary],
    ls_summaries: Dict[int, LSSummary],
    mode: str,
    coverage_note: str,
) -> None:
    parts: List[str] = []
    parts.append(
        "<html><head><meta charset='utf-8'><title>Signal Analysis Dashboard</title>"
    )
    parts.append(
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;} h2{margin-top:30px;} table{border-collapse:collapse;margin:10px 0;} td,th{border:1px solid #ddd;padding:6px 10px;} .row{display:flex;gap:16px;flex-wrap:wrap;} .card{border:1px solid #eee;padding:10px;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.06);} img{max-width:100%;height:auto;border:1px solid #eee;border-radius:4px;} .kpi{display:grid;grid-template-columns:repeat(4, minmax(140px,1fr));gap:8px;} .kpi div{background:#fafafa;border:1px solid #eee;border-radius:6px;padding:8px;} .footer{color:#888;font-size:12px;margin-top:30px;} .banner{padding:10px;border-left:4px solid #3b82f6;background:#eef6ff;margin:10px 0;border-radius:4px;}</style>"
    )
    parts.append("</head><body>")
    parts.append("<h1>Signal Analysis Dashboard</h1>")
    parts.append(
        f"<div class='banner'><b>Mode:</b> {html_escape(mode.upper())}. {html_escape(coverage_note)}</div>"
    )
    parts.append(
        f"<p>Generated: {html_escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>"
    )

    for k in sorted(horizons):
        ics = ic_summaries.get(k)
        lss = ls_summaries.get(k)
        parts.append(f"<h2>Horizon: {k}-day</h2>")
        parts.append("<div class='kpi'>")
        parts.append(
            f"<div><b>IC mean</b><br>{fmt(ics.mean,4) if ics else 'N/A'}</div>"
        )
        parts.append(f"<div><b>IC std</b><br>{fmt(ics.std,4) if ics else 'N/A'}</div>")
        parts.append(f"<div><b>IR</b><br>{fmt(ics.ir,3) if ics else 'N/A'}</div>")
        parts.append(
            f"<div><b>Pos IC ratio</b><br>{fmt(ics.pos_ratio*100,2)+'%' if (ics and not np.isnan(ics.pos_ratio)) else 'N/A'}</div>"
        )
        parts.append(
            f"<div><b>t (norm)</b><br>{fmt(ics.t_stat_norm,2) if ics else 'N/A'}</div>"
        )
        parts.append(
            f"<div><b>p (norm)</b><br>{fmt(ics.p_value_norm,4) if ics else 'N/A'}</div>"
        )
        parts.append(f"<div><b>IC days</b><br>{ics.n_days if ics else 'N/A'}</div>")
        if lss:
            parts.append(f"<div><b>LS mean</b><br>{fmt(lss.mean,6)}</div>")
            parts.append(f"<div><b>LS std</b><br>{fmt(lss.std,6)}</div>")
            parts.append(f"<div><b>LS IR</b><br>{fmt(lss.ir,3)}</div>")
            parts.append(
                f"<div><b>LS pos ratio</b><br>{fmt(lss.pos_ratio*100,2)+'%' if not np.isnan(lss.pos_ratio) else 'N/A'}</div>"
            )
            parts.append(f"<div><b>LS cum</b><br>{fmt(lss.cum_return,4)}</div>")
            parts.append(f"<div><b>LS t (norm)</b><br>{fmt(lss.t_stat_norm,2)}</div>")
            parts.append(f"<div><b>LS p (norm)</b><br>{fmt(lss.p_value_norm,4)}</div>")
        parts.append("</div>")

        parts.append("<div class='row'>")
        parts.append(
            f"<div class='card'><h3>Pred vs GT (Rank-Normalized)</h3><img src='{k}d/rank_scatter.png' alt='Rank-normalized scatter'></div>"
        )
        parts.append(
            f"<div class='card'><h3>Bin Avg Alpha</h3><img src='{k}d/bins_bar.png' alt='Bin average alpha'></div>"
        )
        parts.append(
            f"<div class='card'><h3>Top-K Lift</h3><img src='{k}d/lift_curve.png' alt='Top-K Lift'></div>"
        )
        parts.append("</div>")

    parts.append("<div class='footer'>")
    parts.append(
        "<p>Notes: IC uses Spearman rank correlation. In sparse cross-sections, the analysis switches to global mode with rolling IC from daily-mean series and global rank-binning.</p>"
    )
    parts.append("</div>")

    parts.append("</body></html>")

    with open(os.path.join(out_root, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ---------------------------- runner -----------------------------


def write_summary_csv(path: str, summary: dict) -> None:
    pd.DataFrame([summary]).to_csv(path, index=False)


def analyze(
    csv_path: str,
    out_dir: str,
    bins: int = 20,
    mode: str = "auto",
    roll_window: int = 60,
    date_col: str = "test_date",
    instrument_col: str = "instrument_name",
    only_date: Optional[str] = None,
) -> None:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    _ensure_dir(out_dir)

    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}' in CSV")
    df[date_col] = pd.to_datetime(df[date_col])
    date_filter_note = ""
    if only_date:
        try:
            target_date = pd.to_datetime(only_date)
        except Exception:
            raise ValueError(f"Invalid --date value: {only_date}")
        df = df[df[date_col] == target_date]
        date_filter_note = f" Date filter: {target_date.date()}"

    horizons = detect_horizons(df.columns.tolist())
    if not horizons:
        raise ValueError(
            "No horizons detected. Expect columns like pred_return_1_days and gt_return_1_days."
        )

    # Coverage diagnostics
    counts = cross_section_counts(
        df[[date_col, instrument_col]].drop_duplicates(), date_col, instrument_col
    )
    counts.to_csv(os.path.join(out_dir, "cross_section_counts.csv"), index=False)
    share_ge2 = float((counts["n_instruments"] >= 2).mean()) if len(counts) > 0 else 0.0
    avg_n = float(counts["n_instruments"].mean()) if len(counts) > 0 else 0.0

    chosen_mode = mode
    if mode == "auto":
        num_ge2 = int((counts["n_instruments"] >= 2).sum()) if len(counts) > 0 else 0
        if only_date:
            # Single-date decision
            chosen_mode = "cross" if (num_ge2 >= 1 and len(counts) == 1) else "global"
        else:
            # If at least half of dates have 2+ instruments and we have >= 5 such dates, use cross-sectional
            if share_ge2 >= 0.5 and num_ge2 >= 5:
                chosen_mode = "cross"
            # Otherwise, if there exists at least one valid cross-section, still prefer cross
            elif num_ge2 >= 1:
                chosen_mode = "cross"
            else:
                chosen_mode = "global"

    coverage_note = (
        f"Avg instruments/day: {avg_n:.2f}. Dates with >=2 instruments: {share_ge2:.0%}."
        + date_filter_note
    )

    ic_summaries: Dict[int, ICSummary] = {}
    ls_summaries: Dict[int, LSSummary] = {}

    for k in horizons:
        pred_col = f"pred_return_{k}_days"
        gt_col = f"gt_return_{k}_days"
        sub = df[[date_col, instrument_col, pred_col, gt_col]].copy()

        k_out = os.path.join(out_dir, f"{k}d")
        _ensure_dir(k_out)

        if chosen_mode == "cross":
            # IC per date
            ic_daily = compute_ic_daily(sub, pred_col, gt_col, date_col)
            ic_summary = summarize_ic(ic_daily, k)

            # Bins per date
            bins_daily, bins_mean, counts_daily, ls_daily = compute_bins_daily(
                sub, pred_col, gt_col, date_col, bins
            )
            ls_summary = summarize_ls(ls_daily, k)
        else:
            # Global mode
            d = sub[[date_col, pred_col, gt_col]].dropna().copy()
            # overall IC across all rows
            overall_ic = _spearman_no_scipy(d[pred_col], d[gt_col])
            ic_daily = pd.DataFrame({date_col: [], "ic": []})
            # Rolling IC from daily-mean series (Pearson of ranks)
            daily_means = d.groupby(date_col)[[pred_col, gt_col]].mean().sort_index()
            if len(daily_means) >= 2:
                x = daily_means[pred_col]
                y = daily_means[gt_col]
                # rank-transform then rolling Pearson
                xr = x.rank(method="average")
                yr = y.rank(method="average")
                rc = xr.rolling(window=min(roll_window, len(xr)), min_periods=2).corr(
                    yr
                )
                ic_daily = (
                    rc.rename("ic").reset_index().rename(columns={"level_0": date_col})
                )
            ic_summary = ICSummary(
                k,
                int(ic_daily["ic"].notna().sum()) if not ic_daily.empty else 0,
                float(overall_ic) if overall_ic is not None else np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

            # Global bins
            bins_daily, bins_mean, counts_daily, ls_daily = compute_bins_global(
                sub, pred_col, gt_col, date_col, bins
            )
            ls_summary = summarize_ls(ls_daily, k)

        # Save CSVs
        ic_daily.to_csv(os.path.join(k_out, "ic_daily.csv"), index=False)
        write_summary_csv(
            os.path.join(k_out, "ic_summary.csv"),
            {
                "horizon": k,
                "n_days": ic_summary.n_days,
                "mean": ic_summary.mean,
                "std": ic_summary.std,
                "ir": ic_summary.ir,
                "t_stat_norm": ic_summary.t_stat_norm,
                "p_value_norm": ic_summary.p_value_norm,
                "pos_ratio": ic_summary.pos_ratio,
            },
        )
        bins_daily.to_csv(os.path.join(k_out, "bins_daily.csv"), index_label="date")
        counts_daily.to_csv(
            os.path.join(k_out, "bins_counts_daily.csv"), index_label="date"
        )
        bins_mean.to_csv(os.path.join(k_out, "bins_mean.csv"))
        ls_daily.to_csv(os.path.join(k_out, "long_short_daily.csv"), index=False)
        write_summary_csv(
            os.path.join(k_out, "long_short_summary.csv"),
            {
                "horizon": k,
                "n_days": ls_summary.n_days,
                "mean": ls_summary.mean,
                "std": ls_summary.std,
                "ir": ls_summary.ir,
                "t_stat_norm": ls_summary.t_stat_norm,
                "p_value_norm": ls_summary.p_value_norm,
                "pos_ratio": ls_summary.pos_ratio,
                "cum_return": ls_summary.cum_return,
            },
        )

        # Plots
        # plot_ic_timeseries removed per request
        plot_rank_scatter(
            sub,
            pred_col,
            gt_col,
            os.path.join(k_out, "rank_scatter.png"),
            f"Pred vs GT (Rank-Normalized) ({k}D)",
        )
        plot_bins_bar(
            bins_mean, os.path.join(k_out, "bins_bar.png"), f"Average Alpha by Bin ({k}D)"
        )
        plot_lift_curve(
            sub, pred_col, gt_col, os.path.join(k_out, "lift_curve.png"), f"Top-K Lift ({k}D)"
        )
        # long-short curve figure removed per request

        ic_summaries[k] = ic_summary
        ls_summaries[k] = ls_summary

    build_dashboard(
        out_dir, horizons, ic_summaries, ls_summaries, chosen_mode, coverage_note
    )

    print(f"Done. Results saved to: {out_dir}")
    print(f"Open {os.path.join(out_dir, 'index.html')} in a browser.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Signal analysis dashboard (IC + bins)")
    p.add_argument("--csv", required=True, help="Path to input CSV")
    p.add_argument("--out", default="./signal_analysis_out", help="Output directory")
    p.add_argument("--bins", type=int, default=20, help="Number of bins for analysis")
    p.add_argument(
        "--mode",
        choices=["auto", "cross", "global"],
        default="auto",
        help="Analysis mode",
    )
    p.add_argument(
        "--roll", type=int, default=60, help="Rolling window for global IC time series"
    )
    p.add_argument("--date-col", default="test_date", help="Date column name")
    p.add_argument(
        "--instrument-col", default="instrument_name", help="Instrument column name"
    )
    p.add_argument(
        "--date",
        dest="date",
        default=None,
        help="Analyze a single date (e.g., 2025-08-20)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    analyze(
        csv_path=args.csv,
        out_dir=args.out,
        bins=args.bins,
        mode=args.mode,
        roll_window=args.roll,
        date_col=args.date_col,
        instrument_col=args.instrument_col,
        only_date=args.date,
    )
