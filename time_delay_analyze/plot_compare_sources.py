#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import Optional, List
from analyzers import SourceComparatorLocal


def plot_compare(
    data_dir: Path,
    output_path: Path,
    show: bool = False,
    lag_minutes: int = 0,
) -> None:
    comparator = SourceComparatorLocal(data_dir)
    comparator.plot(output_path, show, lag_minutes=lag_minutes)


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
    parser.add_argument(
        "--lag-minutes",
        type=int,
        default=0,
        help="为CDA曲线应用的时延（分钟，正值表示CDA更晚）",
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
            lag_minutes=int(args.lag_minutes or 0),
        )
    except Exception as exc:
        print(f"[错误] 绘图失败: {exc}", file=sys.stderr)
        return 1

    print(f"[完成] 已保存图像: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


