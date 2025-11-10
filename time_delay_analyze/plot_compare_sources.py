#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import DateFormatter, AutoDateLocator


def find_file_by_suffix(data_dir: Path, suffix: str) -> Optional[Path]:
    """
    Find the first CSV file in data_dir whose name ends with the given suffix (before .csv).
    Example suffix: 'BGSEc_0' or 'Bx'
    """
    candidates: List[Path] = sorted(data_dir.glob(f"*_{suffix}.csv"))
    if candidates:
        return candidates[0]
    return None


def load_series(file_path: Path, time_col: str, value_col: str) -> pd.DataFrame:
    """
    Load a CSV with 'Time' and a single value column.
    Returns a DataFrame with columns ['Time', 'Value'] sorted by time.
    """
    df = pd.read_csv(file_path)
    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"文件列缺失: 需要[{time_col}, {value_col}]，实际为 {list(df.columns)}; 文件: {file_path}")
    df = df[[time_col, value_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df.sort_values(time_col, inplace=True)
    df.rename(columns={time_col: "Time", value_col: "Value"}, inplace=True)
    # 去除无效值
    df = df[pd.notnull(df["Value"])]
    return df


def plot_compare(
    data_dir: Path,
    output_path: Path,
    show: bool = False,
) -> None:
    """
    Plot three subplots: (Bx, By, Bz) comparing CDAWeb and OMNI sources.
    CDAWeb uses columns BGSEc_0/1/2; OMNI uses Bx/By/Bz.
    """
    # 定义三组参数映射： (CDA列名, OMNI列名, 轴标签)
    param_pairs: List[Tuple[str, str, str]] = [
        ("BGSEc_0", "Bx", "Bx (GSE) [nT]"),
        ("BGSEc_1", "By_GSE", "By (GSE) [nT]"),
        ("BGSEc_2", "Bz_GSE", "Bz (GSE) [nT]"),
    ]

    # 收集可用文件并加载
    series_map: Dict[str, Dict[str, pd.DataFrame]] = {}  # key: component, value: {source: df}
    for cda_col, omni_col, _ylabel in param_pairs:
        cda_file = find_file_by_suffix(data_dir, cda_col)
        omni_file = find_file_by_suffix(data_dir, omni_col)

        if cda_file is None:
            raise FileNotFoundError(f"未找到CDAWeb数据文件 (*_{cda_col}.csv) 于 {data_dir}")
        if omni_file is None:
            raise FileNotFoundError(f"未找到OMNI数据文件 (*_{omni_col}.csv) 于 {data_dir}")

        cda_df = load_series(cda_file, "Time", cda_col)
        omni_df = load_series(omni_file, "Time", omni_col)
        series_map[omni_col] = {
            "CDAWeb": cda_df,
            "OMNI": omni_df,
        }

    # 字体设为 Times
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False

    # 生成图形
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharex=True)

    colors = {"CDAWeb": "tab:blue", "OMNI": "tab:orange"}
    styles = {"CDAWeb": "-", "OMNI": "-"}
    labels = {"CDAWeb": "CDAWeb", "OMNI": "OMNI"}

    for ax, (cda_col, omni_col, ylabel) in zip(axes, param_pairs):
        comp = omni_col  # "Bx"|"By"|"Bz"
        comp_map = series_map[comp]

        # 画CDAWeb
        cda_df = comp_map["CDAWeb"]
        ax.plot(
            cda_df["Time"],
            cda_df["Value"],
            styles["CDAWeb"],
            color=colors["CDAWeb"],
            linewidth=0.8,
            alpha=0.9,
            label=f"{labels['CDAWeb']} {cda_col}",
        )
        # 画OMNI
        omni_df = comp_map["OMNI"]
        ax.plot(
            omni_df["Time"],
            omni_df["Value"],
            styles["OMNI"],
            color=colors["OMNI"],
            linewidth=1.0,
            alpha=0.9,
            label=f"{labels['OMNI']} {omni_col}",
        )

        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9, frameon=False)

    axes[-1].set_xlabel("Time (UTC)")

    # x轴时间格式
    locator = AutoDateLocator(minticks=6, maxticks=12)
    formatter = DateFormatter("%H:%M")
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    # 横轴紧致：按两来源共同的时间范围限制，并去掉左右边距
    try:
        # 取三个分量的交集时间范围
        ranges = []
        for _cda, omni_col, _yl in param_pairs:
            comp_map = series_map[omni_col]
            t_omni = comp_map["OMNI"]["Time"]
            t_cda = comp_map["CDAWeb"]["Time"]
            if not t_omni.empty and not t_cda.empty:
                t0 = max(t_omni.min(), t_cda.min())
                t1 = min(t_omni.max(), t_cda.max())
                if t0 < t1:
                    ranges.append((t0, t1))
        if ranges:
            x0 = max(r[0] for r in ranges)
            x1 = min(r[1] for r in ranges)
            if x0 < x1:
                for ax in axes:
                    ax.set_xlim(x0, x1)
                    ax.set_xmargin(0)
                    ax.margins(x=0)
    except Exception:
        pass

    # 标题（尝试从文件名推断日期）
    title_date: Optional[str] = None
    # 尝试优先从OMNI Bx文件名里获得日期片段
    bx_file = find_file_by_suffix(data_dir, "Bx")
    if bx_file:
        # 例: space_weather_omniweb_1min_20151016_Bx.csv -> 20151016
        parts = bx_file.stem.split("_")
        for p in parts:
            if p.isdigit() and len(p) == 8:
                title_date = f"{p[:4]}-{p[4:6]}-{p[6:8]}"
                break
    fig.suptitle(f"CDAWeb vs OMNI Magnetic Field Components{f' ({title_date})' if title_date else ''}", y=0.98, fontsize=14)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="比较CDAWeb与OMNI三分量磁场（同图三子图叠加）")
    default_data_dir = Path(__file__).resolve().parent / "data"
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help=f"数据目录，默认: {default_data_dir}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "compare_cdaweb_omni.png",
        help="输出PNG路径，默认保存在脚本目录",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="显示窗口（同时也会保存PNG）",
    )
    args = parser.parse_args(argv)

    if not args.data_dir.exists():
        print(f"[错误] 数据目录不存在: {args.data_dir}", file=sys.stderr)
        return 2

    try:
        plot_compare(
            data_dir=args.data_dir,
            output_path=args.output,
            show=bool(args.show),
        )
    except Exception as exc:
        print(f"[错误] 绘图失败: {exc}", file=sys.stderr)
        return 1

    print(f"[完成] 已保存图像: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


