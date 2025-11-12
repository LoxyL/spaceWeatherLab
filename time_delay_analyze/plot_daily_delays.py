#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import DateFormatter, AutoDateLocator


def set_times_font():
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False


def tighten_x_axis(axes, x: pd.Series, ycols: List[str], df: pd.DataFrame):
    try:
        # 仅考虑至少一个分量不为 NaN 的日期范围
        mask_any = np.zeros(len(df), dtype=bool)
        for col in ycols:
            if col in df.columns:
                mask_any |= pd.to_numeric(df[col], errors='coerce').notna().values
        if mask_any.any():
            x_valid = pd.to_datetime(df.loc[mask_any, 'Date'])
            x0, x1 = x_valid.min(), x_valid.max()
            if pd.notnull(x0) and pd.notnull(x1) and x0 < x1:
                for ax in axes:
                    ax.set_xlim(x0, x1)
                    ax.set_xmargin(0)
                    ax.margins(x=0)
        # 设定时间刻度格式
        locator = AutoDateLocator(minticks=6, maxticks=10)
        formatter = DateFormatter("%Y-%m-%d")
        for ax in axes:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
    except Exception:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="从 CSV 绘制每日最佳时延折线图与总体直方图")
    parser.add_argument("--csv", type=Path, required=True, help="daily_best_lags.csv 路径")
    parser.add_argument("--out-line", type=Path, default=Path(__file__).resolve().parent / "daily_best_lags.png",
                        help="输出每日时延折线图")
    parser.add_argument("--out-hist", type=Path, default=Path(__file__).resolve().parent / "daily_lag_hist.png",
                        help="输出总体时延直方图")
    parser.add_argument("--show", action="store_true", help="显示图像窗口")
    args = parser.parse_args(argv)

    set_times_font()

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"[错误] 读取 CSV 失败: {e}", file=sys.stderr)
        return 2

    if 'Date' not in df.columns:
        print("[错误] CSV 缺少 'Date' 列。", file=sys.stderr)
        return 2

    # 折线图（3子图），横轴紧
    try:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharex=True)
        x = pd.to_datetime(df['Date'])
        series = [("Bx_lag", "Bx delay (min)"), ("By_GSE_lag", "By (GSE) delay (min)"), ("Bz_GSE_lag", "Bz (GSE) delay (min)")]
        for ax, (col, ylabel) in zip(axes, series):
            y = pd.to_numeric(df.get(col, pd.Series([])), errors='coerce')
            ax.plot(x, y, marker='o', linewidth=1.2, color='tab:blue', alpha=0.9)
            ax.axhline(0.0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.3)
        axes[-1].set_xlabel("Date (UTC)")
        tighten_x_axis(axes, x, ["Bx_lag", "By_GSE_lag", "Bz_GSE_lag"], df)
        fig.tight_layout()
        args.out_line.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_line, dpi=150)
        if args.show:
            plt.show()
        else:
            plt.close(fig)
        print(f"[OK] 已保存每日时延折线图: {args.out_line}")
    except Exception as e:
        print(f"[WARN] 绘制每日折线图失败: {e}", file=sys.stderr)

    # 总体直方图（3子图）
    try:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), sharex=False)
        series = [("Bx_lag", "Bx delay (min)"), ("By_GSE_lag", "By (GSE) delay (min)"), ("Bz_GSE_lag", "Bz (GSE) delay (min)")]
        for ax, (col, xlabel) in zip(axes, series):
            vals = pd.to_numeric(df.get(col, pd.Series([])), errors='coerce').dropna()
            if len(vals) > 0:
                bins = np.arange(np.nanmin(vals)-0.5, np.nanmax(vals)+1.5, 1.0)
                ax.hist(vals, bins=bins, color='tab:orange', alpha=0.85, edgecolor='white')
            ax.axvline(0.0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.grid(True, linestyle='--', alpha=0.3)
        fig.tight_layout()
        args.out_hist.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_hist, dpi=150)
        if args.show:
            plt.show()
        else:
            plt.close(fig)
        print(f"[OK] 已保存总体时延直方图: {args.out_hist}")
    except Exception as e:
        print(f"[WARN] 绘制直方图失败: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


