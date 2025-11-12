#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import Optional, List
from analyzers import SourceComparatorUsingPkg


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="使用 space_weather_data 包按时间抓取并比较 CDAWeb 与 OMNI（Bx/By/Bz，GSE）")
    parser.add_argument("--time", required=True, help="时间输入（如：2023、2023-06、2023-06-15、2023-06-15/2023-06-16）")
    parser.add_argument("--dataset", default="AC_H0_MFI", help="CDAWeb 数据集（默认：AC_H0_MFI，可选 WI_H0_MFI 等）")
    parser.add_argument("--cdaweb-datatype", default=None, help="CDAWeb 数据类型（如 h0/h3；默认自动推断）")
    parser.add_argument("--omni-resolution", default="1min", choices=["1min", "5min", "hourly"], help="OMNI 时间分辨率（默认 1min）")
    parser.add_argument("--pkg-dir", type=Path, default=Path(__file__).resolve().parent.parent / "space_weather_data", help="space_weather_data 包目录路径")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "compare_pkg_cda_omni.png", help="输出PNG路径")
    parser.add_argument("--show", action="store_true", help="显示窗口")
    parser.add_argument("--lag-minutes", type=int, default=0, help="为CDA曲线应用的时延（分钟，正值表示CDA更晚）")
    args = parser.parse_args(argv)

    try:
        comparator = SourceComparatorUsingPkg(
            pkg_dir=args.pkg_dir,
            dataset=args.dataset,
            cdaweb_datatype=args.cdaweb_datatype,
            omni_resolution=args.omni_resolution,
        )
        comparator.plot(time_str=args.time, output=args.output, show=args.show, lag_minutes=int(args.lag_minutes or 0))
        print(f"[完成] 已保存图像: {args.output}")
        return 0
    except Exception as exc:
        print(f"[错误] 运行失败：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


