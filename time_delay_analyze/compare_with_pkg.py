#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import DateFormatter, AutoDateLocator


def add_pkg_path(pkg_dir: Path) -> None:
    """
    Ensure space_weather_data modules can be imported without modifying that package.
    We add its directory to sys.path so its absolute imports (e.g., `from config import ...`)
    resolve to the package-local files.
    """
    pkg_dir = pkg_dir.resolve()
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))


def fetch_data_with_pkg(
    time_str: str,
    dataset: str,
    cdaweb_datatype: Optional[str],
    omni_resolution: str,
    pkg_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """
    Use space_weather_data tools to fetch OMNI and CDAWeb data for Bx/By/Bz (GSE).
    Returns a dict: {'omni': df, 'cda': df}
    """
    add_pkg_path(pkg_dir)
    from time_parser import TimeParser
    from data_fetcher import DataFetcher

    parser = TimeParser()
    start_dt, end_dt = parser.parse(time_str)

    fetcher = DataFetcher(resolution=omni_resolution)

    # OMNI（标准化列名：Bx, By_GSE, Bz_GSE）
    df_omni = fetcher.fetch_omni(start_dt, end_dt)
    # 仅保留需要的列
    keep_omni = ["Time", "Bx", "By_GSE", "Bz_GSE"]
    df_omni = df_omni[[c for c in keep_omni if c in df_omni.columns]].copy()

    # CDA（请求分量：BX_GSE, BY_GSE, BZ_GSE；最终列名与请求一致）
    cda_params = ["BX_GSE", "BY_GSE", "BZ_GSE"]
    df_cda = fetcher.fetch_cdaweb(dataset, start_dt, end_dt, cda_params, datatype=cdaweb_datatype)
    df_cda = df_cda[[c for c in ["Time"] + cda_params if c in df_cda.columns]].copy()

    return {"omni": df_omni, "cda": df_cda}


def plot_compare(omni: pd.DataFrame, cda: pd.DataFrame, output: Path, show: bool = False) -> None:
    """
    Draw three subplots: Bx, By_GSE, Bz_GSE; overlay OMNI vs CDA.
    """
    # 字体设为 Times
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False

    # 清洗时间
    for df in (omni, cda):
        if "Time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Time"]):
            df["Time"] = pd.to_datetime(df["Time"])
        df.sort_values("Time", inplace=True)

    components = [
        ("Bx", "Bx (GSE) [nT]"),
        ("By_GSE", "By (GSE) [nT]"),
        ("Bz_GSE", "Bz (GSE) [nT]"),
    ]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharex=True)
    colors = {"OMNI": "tab:orange", "CDAWeb": "tab:blue"}

    for ax, (col, ylabel) in zip(axes, components):
        # OMNI
        if col in omni.columns:
            ax.plot(omni["Time"], pd.to_numeric(omni[col], errors="coerce"),
                    color=colors["OMNI"], linewidth=1.0, alpha=0.9, label=f"OMNI {col}")
        # CDA
        cda_col = {"Bx": "BX_GSE", "By_GSE": "BY_GSE", "Bz_GSE": "BZ_GSE"}[col]
        if cda_col in cda.columns:
            ax.plot(cda["Time"], pd.to_numeric(cda[cda_col], errors="coerce"),
                    color=colors["CDAWeb"], linewidth=0.8, alpha=0.9, label=f"CDAWeb {cda_col}")

        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        if col in ("Bx", "By_GSE", "Bz_GSE"):
            ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8, alpha=0.4)

    axes[-1].set_xlabel("Time (UTC)")

    locator = AutoDateLocator(minticks=6, maxticks=12)
    formatter = DateFormatter("%m-%d %H:%M")
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    # 横轴紧致：优先使用两来源的共同时间范围，并去掉左右边距
    all_times = pd.concat([omni["Time"].dropna(), cda["Time"].dropna()], axis=0)
    try:
        if not omni["Time"].dropna().empty and not cda["Time"].dropna().empty:
            t0 = max(omni["Time"].min(), cda["Time"].min())
            t1 = min(omni["Time"].max(), cda["Time"].max())
            if t0 < t1:
                for ax in axes:
                    ax.set_xlim(t0, t1)
                    ax.set_xmargin(0)
                    ax.margins(x=0)
    except Exception:
        pass

    # 标题区间（使用绘制范围或联合范围）
    title = "CDAWeb vs OMNI Magnetic Field (GSE)"
    try:
        xlims = axes[-1].get_xlim()
        # 将浮点日期转换为时间戳格式
        import matplotlib.dates as mdates
        t0_dt = mdates.num2date(xlims[0])
        t1_dt = mdates.num2date(xlims[1])
        title += f"  [{t0_dt.strftime('%Y-%m-%d %H:%M')} — {t1_dt.strftime('%Y-%m-%d %H:%M')}]"
    except Exception:
        if not all_times.empty:
            t0, t1 = all_times.min(), all_times.max()
            title += f"  [{t0.strftime('%Y-%m-%d %H:%M')} — {t1.strftime('%Y-%m-%d %H:%M')}]"
    fig.suptitle(title, y=0.98, fontsize=14)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="使用 space_weather_data 包按时间抓取并比较 CDAWeb 与 OMNI（Bx/By/Bz，GSE）")
    parser.add_argument("--time", required=True, help="时间输入（如：2023、2023-06、2023-06-15、2023-06-15/2023-06-16）")
    parser.add_argument("--dataset", default="AC_H0_MFI", help="CDAWeb 数据集（默认：AC_H0_MFI，可选 WI_H0_MFI 等）")
    parser.add_argument("--cdaweb-datatype", default=None, help="CDAWeb 数据类型（如 h0/h3；默认自动推断）")
    parser.add_argument("--omni-resolution", default="1min", choices=["1min", "5min", "hourly"], help="OMNI 时间分辨率（默认 1min）")
    parser.add_argument("--pkg-dir", type=Path, default=Path(__file__).resolve().parent.parent / "space_weather_data", help="space_weather_data 包目录路径")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "compare_pkg_cda_omni.png", help="输出PNG路径")
    parser.add_argument("--show", action="store_true", help="显示窗口")
    args = parser.parse_args(argv)

    try:
        data = fetch_data_with_pkg(
            time_str=args.time,
            dataset=args.dataset,
            cdaweb_datatype=args.cdaweb_datatype,
            omni_resolution=args.omni_resolution,
            pkg_dir=args.pkg_dir,
        )
        if data["omni"].empty and data["cda"].empty:
            print("[错误] 未获取到任何数据。")
            return 1
        plot_compare(data["omni"], data["cda"], output=args.output, show=args.show)
        print(f"[完成] 已保存图像: {args.output}")
        return 0
    except Exception as exc:
        print(f"[错误] 运行失败：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


