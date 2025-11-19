#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import sys
import pandas as pd
import numpy as np

from analyzers import SourceComparatorUsingPkg, TimeDelayAnalyzer


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
    parser = argparse.ArgumentParser(description="按天计算 OMNI 与 CDA 的最佳时延（Bx/By_GSE/Bz_GSE 或 太阳风速度 Vsw↔Vp），仅导出CSV")
    parser.add_argument("--time", required=True, help="时间范围（如 2015-10-14/2015-10-21 或单日 2015-10-14）")
    parser.add_argument("--pkg-dir", type=Path, default=Path(__file__).resolve().parent.parent / "space_weather_data",
                        help="space_weather_data 包目录")
    parser.add_argument("--dataset", default="AC_H0_MFI", help="CDAWeb 数据集（默认 AC_H0_MFI）")
    parser.add_argument("--cdaweb-datatype", default=None, help="CDAWeb 数据类型（如 h0/h3；默认自动推断）")
    parser.add_argument("--omni-resolution", default="1min", choices=["1min", "5min", "hourly"], help="OMNI 分辨率")
    parser.add_argument("--out-csv", type=Path, default=Path(__file__).resolve().parent / "daily_best_lags.csv",
                        help="输出每日最佳时延 CSV 路径")
    parser.add_argument("--overwrite", action="store_true", help="强制重新下载远端数据（忽略本地CDF缓存）")
    parser.add_argument("--verbose", action="store_true", help="输出诊断信息")
    parser.add_argument("--mode", choices=["mag", "speed", "both"], default="both",
                        help="计算模式：mag=仅磁场；speed=仅速度（Vsw↔Vp）；both=两者都算（默认）")
    args = parser.parse_args(argv)

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
    # 是否计算磁场/速度
    do_mag = args.mode in ("mag", "both")
    do_speed = args.mode in ("speed", "both")

    components = [("Bx", "BX_GSE"), ("By_GSE", "BY_GSE"), ("Bz_GSE", "BZ_GSE")] if do_mag else []
    # 速度分量（如可用：OMNI Vsw 与 ACE SWE Vp）
    speed_pair = ("Vsw", "Vp")

    for day in days:
        day_str = day.strftime('%Y-%m-%d')
        try:
            data = comparator.fetch(day_str, overwrite=args.overwrite, speed_only=(args.mode == "speed"))
            omni = data.get("omni", pd.DataFrame())
            cda = data.get("cda", pd.DataFrame())
            if omni is None or omni.empty or cda is None or cda.empty:
                if args.verbose:
                    print(f"[INFO] {day_str}: 数据为空，跳过。")
                empty_row = {
                    "Date": day_str,
                }
                if do_mag:
                    empty_row.update({
                        "Bx_lag": np.nan, "By_GSE_lag": np.nan, "Bz_GSE_lag": np.nan,
                        "Bx_mse": np.nan, "By_GSE_mse": np.nan, "Bz_GSE_mse": np.nan,
                        "Vsw_mean": np.nan, "theory_lag_min": np.nan,
                    })
                if do_speed:
                    empty_row.update({"Vsw_lag": np.nan, "Vsw_mse": np.nan})
                records.append(empty_row)
                continue

            row = {"Date": day_str}
            if do_mag:
                for omni_col, cda_col in components:
                    res = analyzer.analyze_component(omni, cda, omni_col=omni_col, cda_col=cda_col,
                                                     lag_min=-120, lag_max=120)
                    row[f"{omni_col}_lag"] = res.get("best_lag")
                    row[f"{omni_col}_mse"] = res.get("best_mse")
                # 追加：保存 OMNI 日内平均速度，并计算更合理的理论传播时延
                try:
                    # 选速度优先级：|Vx| > Vsw（单位 km/s）
                    v_use = None
                    if "Vx" in omni.columns:
                        v_use = pd.to_numeric(omni["Vx"], errors="coerce").abs()
                    elif "Vsw" in omni.columns:
                        v_use = pd.to_numeric(omni["Vsw"], errors="coerce")
                    # 密度 n [cm^-3]
                    n_use = pd.to_numeric(omni["nsw"], errors="coerce") if "nsw" in omni.columns else None

                    if v_use is not None and v_use.notna().any():
                        # 速度太小/异常的剔除
                        v = v_use.replace([np.inf, -np.inf], np.nan)
                        v[v <= 50.0] = np.nan  # 过滤极小速度
                        mean_v = float(np.nanmean(v))
                        row["Vsw_mean"] = mean_v if np.isfinite(mean_v) else np.nan

                        # 动压估计（若有 n）：Pdyn[nPa] = 1.6726e-6 * n[cm^-3] * V[km/s]^2
                        if n_use is not None and n_use.notna().any():
                            n = n_use.replace([np.inf, -np.inf], np.nan)
                            Pdyn = 1.6726e-6 * n * (v**2)
                            # 经验式弓激波鼻部距离（单位 Re）：~ 14 * Pdyn^(-1/6.6)，限制在 [12, 20] Re
                            Rbsn_Re = 14.0 * (Pdyn ** (-1.0 / 6.6))
                            Rbsn_Re = Rbsn_Re.clip(lower=12.0, upper=20.0)
                            Rbsn_km = Rbsn_Re * 6371.0
                        else:
                            # 无动压时，使用典型常数 15 Re
                            Rbsn_km = pd.Series(15.0 * 6371.0, index=v.index)

                        # L1 标称距离（km）
                        D_L1_km = 1.5e6
                        D_eff = D_L1_km - Rbsn_km
                        # t[min] = D_eff / v / 60
                        t_series = (D_eff / v) / 60.0
                        # 稳健统计：取中位数
                        t_theory = float(np.nanmedian(t_series))
                        row["theory_lag_min"] = t_theory if np.isfinite(t_theory) else np.nan
                    else:
                        row["Vsw_mean"] = np.nan
                        row["theory_lag_min"] = np.nan
                except Exception:
                    row["Vsw_mean"] = np.nan
                    row["theory_lag_min"] = np.nan

            # 可选：速度（仅当两侧列同时存在时计算）
            if do_speed:
                try:
                    omni_speed, cda_speed = speed_pair
                    if (omni_speed in omni.columns) and (cda_speed in cda.columns):
                        res_v = analyzer.analyze_component(omni, cda, omni_col=omni_speed, cda_col=cda_speed,
                                                           lag_min=-120, lag_max=120)
                        row["Vsw_lag"] = res_v.get("best_lag")
                        row["Vsw_mse"] = res_v.get("best_mse")
                    else:
                        row["Vsw_lag"] = np.nan
                        row["Vsw_mse"] = np.nan
                except Exception:
                    row["Vsw_lag"] = np.nan
                    row["Vsw_mse"] = np.nan

            records.append(row)
            if args.verbose:
                parts = [f"Date={day_str}"]
                if do_mag:
                    parts.append(f"Bx={row.get('Bx_lag')}")
                    parts.append(f"By_GSE={row.get('By_GSE_lag')}")
                    parts.append(f"Bz_GSE={row.get('Bz_GSE_lag')}")
                    parts.append(f"Vsw_mean={row.get('Vsw_mean')}")
                    parts.append(f"t_theory={row.get('theory_lag_min')} min")
                if do_speed:
                    parts.append(f"Vsw={row.get('Vsw_lag')}")
                print("[OK] " + "  ".join(parts))
        except Exception as e:
            print(f"[WARN] {day_str} 计算失败: {e}", file=sys.stderr)
            err_row = {
                "Date": day_str,
            }
            if do_mag:
                err_row.update({
                    "Bx_lag": np.nan, "By_GSE_lag": np.nan, "Bz_GSE_lag": np.nan,
                    "Bx_mse": np.nan, "By_GSE_mse": np.nan, "Bz_GSE_mse": np.nan,
                    "Vsw_mean": np.nan, "theory_lag_min": np.nan,
                })
            if do_speed:
                err_row.update({"Vsw_lag": np.nan, "Vsw_mse": np.nan})
            records.append(err_row)

    df = pd.DataFrame.from_records(records)
    try:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"[OK] 已保存每日最佳时延 CSV: {args.out_csv}")
    except Exception as e:
        print(f"[WARN] 保存 CSV 失败: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


