#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import sys
import pandas as pd

from analyzers import SourceComparatorUsingPkg, TimeDelayAnalyzer


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="基于滑动MSE的OMNI与CDA时延分析（1分钟统一分辨率+插值）")
    parser.add_argument("--time", required=True, help="时间输入（如 2015-10-16 或 2015-10-16/2015-10-17）")
    parser.add_argument("--pkg-dir", type=Path, default=Path(__file__).resolve().parent.parent / "space_weather_data",
                        help="space_weather_data 包目录")
    parser.add_argument("--dataset", default="AC_H0_MFI", help="CDAWeb 数据集（默认 AC_H0_MFI）")
    parser.add_argument("--cdaweb-datatype", default=None, help="CDAWeb 数据类型（如 h0/h3；默认自动推断）")
    parser.add_argument("--omni-resolution", default="1min", choices=["1min", "5min", "hourly"], help="OMNI 分辨率")
    parser.add_argument("--lag-min", type=int, default=-120, help="最小滞后（分钟）")
    parser.add_argument("--lag-max", type=int, default=120, help="最大滞后（分钟）")
    parser.add_argument("--out-csv", type=Path, default=Path(__file__).resolve().parent / "mse_curves.csv",
                        help="输出MSE曲线CSV（每个分量一列）")
    parser.add_argument("--out-plot", type=Path, default=Path(__file__).resolve().parent / "mse_curves.png",
                        help="输出MSE曲线图")
    parser.add_argument("--show", action="store_true", help="显示MSE曲线窗口")
    parser.add_argument("--verbose", action="store_true", help="输出详细诊断信息")
    args = parser.parse_args(argv)

    try:
        # 获取数据（不修改包，走 pkg 接口）
        comparator = SourceComparatorUsingPkg(
            pkg_dir=args.pkg_dir, dataset=args.dataset,
            cdaweb_datatype=args.cdaweb_datatype, omni_resolution=args.omni_resolution
        )
        data = comparator.fetch(args.time)
        omni = data["omni"]
        cda = data["cda"]
        if omni.empty or cda.empty:
            print("[错误] 未获取到数据。", file=sys.stderr)
            return 1

        analyzer = TimeDelayAnalyzer()
        components: List[Tuple[str, str]] = [("Bx", "BX_GSE"), ("By_GSE", "BY_GSE"), ("Bz_GSE", "BZ_GSE")]
        results: Dict[str, Dict[str, object]] = {}
        for oc, cc in components:
            if args.verbose:
                # 诊断：列可用性与时间覆盖
                def coverage_info(df: pd.DataFrame, col: str) -> str:
                    tmin = pd.to_datetime(df['Time']).min() if 'Time' in df.columns and len(df) else None
                    tmax = pd.to_datetime(df['Time']).max() if 'Time' in df.columns and len(df) else None
                    nn = pd.to_numeric(df.get(col, pd.Series([])), errors='coerce')
                    nn_count = int(nn.notna().sum()) if len(nn) else 0
                    nn_tmin = pd.to_datetime(df.loc[pd.to_numeric(df.get(col, pd.Series([])), errors='coerce').notna(), 'Time']).min() if nn_count>0 else None
                    nn_tmax = pd.to_datetime(df.loc[pd.to_numeric(df.get(col, pd.Series([])), errors='coerce').notna(), 'Time']).max() if nn_count>0 else None
                    return f"all:[{tmin} — {tmax}] nonnull({nn_count}):[{nn_tmin} — {nn_tmax}]"
                print(f"[VERBOSE] OMNI {oc} -> {coverage_info(omni, oc)}")
                print(f"[VERBOSE] CDA  {cc} -> {coverage_info(cda, cc)}")

            results[oc] = analyzer.analyze_component(
                omni, cda, omni_col=oc, cda_col=cc,
                lag_min=args.lag_min, lag_max=args.lag_max
            )
            if args.verbose:
                curve = results[oc].get('mse_curve')
                if curve is not None and not curve.empty:
                    valid_n = int(curve['n'].max())
                    print(f"[VERBOSE] {oc} curve points: {len(curve)}, max paired samples per lag: {valid_n}")
                else:
                    print(f"[VERBOSE] {oc} curve is empty (no paired samples after alignment)")

        # 输出最佳时延
        print("\nBest lag (minutes) and MSE:")
        for comp in ["Bx", "By_GSE", "Bz_GSE"]:
            r = results.get(comp, {})
            print(f"  {comp:6s}: lag={r.get('best_lag')}  mse={r.get('best_mse')}")

        # 保存MSE曲线CSV（合并为一张表：index为lag分钟，每列为一个分量的mse）
        try:
            merged = None
            for comp in ["Bx", "By_GSE", "Bz_GSE"]:
                curve = results[comp]['mse_curve']
                if curve is None or curve.empty:
                    continue
                df = curve[['lag_min', 'mse']].rename(columns={'mse': comp}).set_index('lag_min')
                merged = df if merged is None else merged.join(df, how='outer')
            if merged is not None:
                args.out_csv.parent.mkdir(parents=True, exist_ok=True)
                merged.to_csv(args.out_csv)
                print(f"[OK] MSE曲线CSV已保存: {args.out_csv}")
        except Exception as e:
            print(f"[WARN] 保存MSE曲线CSV失败: {e}", file=sys.stderr)

        # 绘图
        try:
            analyzer.plot_mse_curves(results, output=args.out_plot, show=args.show)
            print(f"[OK] MSE曲线图已保存: {args.out_plot}")
        except Exception as e:
            print(f"[WARN] 绘图失败: {e}", file=sys.stderr)

        return 0
    except Exception as exc:
        print(f"[错误] 时延分析失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


