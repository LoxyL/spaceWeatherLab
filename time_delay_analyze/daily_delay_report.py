#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from analyzers import SourceComparatorUsingPkg, TimeDelayAnalyzer


def set_times_font():
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False


def parse_time_range(time_str: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if '/' in time_str:
        a, b = time_str.split('/', 1)
        start = pd.to_datetime(a).normalize()
        end = pd.to_datetime(b).normalize()
    else:
        start = pd.to_datetime(time_str).normalize()
        end = start
    if end < start:
        start, end = end, start
    return start, end


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="按天计算 OMNI 与 CDA 的最佳时延（Bx/By_GSE/Bz_GSE），输出每日折线与总体直方图，并保存CSV")
    parser.add_argument("--time", required=True, help="时间范围（如 2015-10-14/2015-10-21 或单日 2015-10-14）")
    parser.add_argument("--pkg-dir", type=Path, default=Path(__file__).resolve().parent.parent / "space_weather_data",
                        help="space_weather_data 包目录")
    parser.add_argument("--dataset", default="AC_H0_MFI", help="CDAWeb 数据集（默认 AC_H0_MFI）")
    parser.add_argument("--cdaweb-datatype", default=None, help="CDAWeb 数据类型（如 h0/h3；默认自动推断）")
    parser.add_argument("--omni-resolution", default="1min", choices=["1min", "5min", "hourly"], help="OMNI 分辨率")
    parser.add_argument("--out-csv", type=Path, default=Path(__file__).resolve().parent / "daily_best_lags.csv",
                        help="输出每日最佳时延 CSV 路径")
    parser.add_argument("--out-line", type=Path, default=Path(__file__).resolve().parent / "daily_best_lags.png",
                        help="输出每日时延折线图")
    parser.add_argument("--out-hist", type=Path, default=Path(__file__).resolve().parent / "daily_lag_hist.png",
                        help="输出总体时延直方图")
    parser.add_argument("--show", action="store_true", help="显示图像窗口")
    parser.add_argument("--verbose", action="store_true", help="输出诊断信息")
    parser.add_argument("--overwrite", action="store_true", help="强制重新下载远端数据（忽略本地CDF缓存）")
    args = parser.parse_args(argv)

    set_times_font()

    start_day, end_day = parse_time_range(args.time)
    days = pd.date_range(start=start_day, end=end_day, freq='D')
    if len(days) == 0:
        print("[错误] 时间范围为空。", file=sys.stderr)
        return 1

    comparator = SourceComparatorUsingPkg(
        pkg_dir=args.pkg_dir, dataset=args.dataset,
        cdaweb_datatype=args.cdaweb_datatype, omni_resolution=args.omni_resolution
    )
    analyzer = TimeDelayAnalyzer()

    records: List[Dict[str, object]] = []
    components = [("Bx", "BX_GSE"), ("By_GSE", "BY_GSE"), ("Bz_GSE", "BZ_GSE")]

    for day in days:
        day_str = day.strftime('%Y-%m-%d')
        try:
            data = comparator.fetch(day_str, overwrite=args.overwrite)
            omni = data.get("omni", pd.DataFrame())
            cda = data.get("cda", pd.DataFrame())
            if omni is None or omni.empty or cda is None or cda.empty:
                if args.verbose:
                    print(f"[INFO] {day_str}: 数据为空，跳过。")
                records.append({"Date": day_str, "Bx_lag": np.nan, "By_GSE_lag": np.nan, "Bz_GSE_lag": np.nan,
                                "Bx_mse": np.nan, "By_GSE_mse": np.nan, "Bz_GSE_mse": np.nan})
                continue

            row = {"Date": day_str}
            for omni_col, cda_col in components:
                res = analyzer.analyze_component(omni, cda, omni_col=omni_col, cda_col=cda_col,
                                                 lag_min=-120, lag_max=120)
                row[f"{omni_col}_lag"] = res.get("best_lag")
                row[f"{omni_col}_mse"] = res.get("best_mse")
            records.append(row)
            if args.verbose:
                print(f"[OK] {day_str}  Bx={row['Bx_lag']}, By_GSE={row['By_GSE_lag']}, Bz_GSE={row['Bz_GSE_lag']}")
        except Exception as e:
            print(f"[WARN] {day_str} 计算失败: {e}", file=sys.stderr)
            records.append({"Date": day_str, "Bx_lag": np.nan, "By_GSE_lag": np.nan, "Bz_GSE_lag": np.nan,
                            "Bx_mse": np.nan, "By_GSE_mse": np.nan, "Bz_GSE_mse": np.nan})

    df = pd.DataFrame.from_records(records)
    # 保存 CSV
    try:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"[OK] 已保存每日最佳时延 CSV: {args.out_csv}")
    except Exception as e:
        print(f"[WARN] 保存 CSV 失败: {e}", file=sys.stderr)

    # 绘图：每日折线（3子图）
    try:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharex=True)
        x = pd.to_datetime(df['Date'])
        series = [("Bx_lag", "Bx delay (min)"), ("By_GSE_lag", "By (GSE) delay (min)"), ("Bz_GSE_lag", "Bz (GSE) delay (min)")]
        for ax, (col, ylabel) in zip(axes, series):
            ax.plot(x, df[col], marker='o', linewidth=1.2, color='tab:blue', alpha=0.9)
            ax.axhline(0.0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.3)
        axes[-1].set_xlabel("Date (UTC)")
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

    # 绘图：总体直方图（3子图）
    try:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), sharex=False)
        series = [("Bx_lag", "Bx delay (min)"), ("By_GSE_lag", "By (GSE) delay (min)"), ("Bz_GSE_lag", "Bz (GSE) delay (min)")]
        for ax, (col, xlabel) in zip(axes, series):
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
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


