#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import DateFormatter, AutoDateLocator

import sys
import datetime as dt


def _set_times_font():
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False


def _tighten_x_axis(axes, ranges: List[Tuple[pd.Timestamp, pd.Timestamp]]):
    try:
        if not ranges:
            return
        x0 = max(r[0] for r in ranges)
        x1 = min(r[1] for r in ranges)
        if x0 < x1:
            for ax in axes:
                ax.set_xlim(x0, x1)
                ax.set_xmargin(0)
                ax.margins(x=0)
    except Exception:
        pass


class ComparisonPlotter:
    """
    统一的绘图封装：
    - plot_from_series: 适用于本地CSV逐列加载后的 series_map 结构
    - plot_from_dataframes: 适用于两张DataFrame（OMNI/CDA）的列名直接对齐
    """
    def __init__(self):
        _set_times_font()

    @staticmethod
    def _format_time_axis(axes):
        locator = AutoDateLocator(minticks=6, maxticks=12)
        formatter = DateFormatter("%m-%d %H:%M")
        for ax in axes:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

    def plot_from_series(
        self,
        series_map: Dict[str, Dict[str, pd.DataFrame]],
        param_pairs: List[Tuple[str, str, str]],
        output_path: Path,
        title: Optional[str] = None,
        show: bool = False,
        lag_minutes: int = 0,
    ) -> None:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharex=True)
        colors = {"CDAWeb": "tab:blue", "OMNI": "tab:orange"}

        for ax, (cda_col, omni_col, ylabel) in zip(axes, param_pairs):
            comp = omni_col
            comp_map = series_map[comp]
            delta = pd.to_timedelta(lag_minutes, unit='m')
            ax.plot(
                comp_map["CDAWeb"]["Time"] + delta, comp_map["CDAWeb"]["Value"],
                "-", color=colors["CDAWeb"], linewidth=0.8, alpha=0.9, label=f"CDAWeb {cda_col}"
            )
            ax.plot(
                comp_map["OMNI"]["Time"], comp_map["OMNI"]["Value"],
                "-", color=colors["OMNI"], linewidth=1.0, alpha=0.9, label=f"OMNI {omni_col}"
            )
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="upper right", fontsize=9, frameon=False)
            ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8, alpha=0.4)

        axes[-1].set_xlabel("Time (UTC)")
        self._format_time_axis(axes)

        # 紧凑横轴（按三个分量交集）
        ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for _cda, omni_col, _yl in param_pairs:
            comp_map = series_map[omni_col]
            t_omni = comp_map["OMNI"]["Time"]
            t_cda = comp_map["CDAWeb"]["Time"] + pd.to_timedelta(lag_minutes, unit='m')
            if not t_omni.empty and not t_cda.empty:
                t0 = max(t_omni.min(), t_cda.min())
                t1 = min(t_omni.max(), t_cda.max())
                if t0 < t1:
                    ranges.append((t0, t1))
        _tighten_x_axis(axes, ranges)

        if title:
            fig.suptitle(title, y=0.98, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_from_dataframes(
        self,
        omni: pd.DataFrame,
        cda: pd.DataFrame,
        components: List[Tuple[str, str, str]],  # (omni_col, ylabel, cda_col)
        output: Path,
        title_prefix: str = "CDAWeb vs OMNI Magnetic Field (GSE)",
        show: bool = False,
        lag_minutes: int = 0,
    ) -> None:
        for df in (omni, cda):
            if "Time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Time"]):
                df["Time"] = pd.to_datetime(df["Time"])
            df.sort_values("Time", inplace=True)

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharex=True)
        colors = {"OMNI": "tab:orange", "CDAWeb": "tab:blue"}
        delta = pd.to_timedelta(lag_minutes, unit='m')

        for ax, (omni_col, ylabel, cda_col) in zip(axes, components):
            if omni_col in omni.columns:
                ax.plot(omni["Time"], pd.to_numeric(omni[omni_col], errors="coerce"),
                        color=colors["OMNI"], linewidth=1.0, alpha=0.9, label=f"OMNI {omni_col}")
            if cda_col in cda.columns:
                ax.plot(cda["Time"] + delta, pd.to_numeric(cda[cda_col], errors="coerce"),
                        color=colors["CDAWeb"], linewidth=0.8, alpha=0.9, label=f"CDAWeb {cda_col}")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="upper right", fontsize=9, frameon=False)
            ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8, alpha=0.4)

        axes[-1].set_xlabel("Time (UTC)")
        self._format_time_axis(axes)

        # 紧凑横轴（两来源共同范围）
        ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        if not omni.empty and not cda.empty and "Time" in omni.columns and "Time" in cda.columns:
            t0 = max(omni["Time"].min(), (cda["Time"] + delta).min())
            t1 = min(omni["Time"].max(), (cda["Time"] + delta).max())
            if t0 < t1:
                ranges.append((t0, t1))
        _tighten_x_axis(axes, ranges)

        # 标题时间段（以xlim为准）
        try:
            import matplotlib.dates as mdates
            x0, x1 = axes[-1].get_xlim()
            t0_dt = mdates.num2date(x0)
            t1_dt = mdates.num2date(x1)
            title = f"{title_prefix}  [{t0_dt.strftime('%Y-%m-%d %H:%M')} — {t1_dt.strftime('%Y-%m-%d %H:%M')}]"
        except Exception:
            title = title_prefix
        fig.suptitle(title, y=0.98, fontsize=14)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)


class SourceComparatorLocal:
    """
    从本地CSV比较 CDAWeb 与 OMNI（默认后缀匹配：BGSEc_0/1/2 与 Bx/By_GSE/Bz_GSE）。
    生成一张图，含三个子图（Bx、By_GSE、Bz_GSE）。
    """
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        _set_times_font()
        self._plotter = ComparisonPlotter()

    @staticmethod
    def _find_file_by_suffix(data_dir: Path, suffix: str) -> Optional[Path]:
        candidates = sorted(data_dir.glob(f"*_{suffix}.csv"))
        if candidates:
            return candidates[0]
        return None

    @staticmethod
    def _load_series(file_path: Path, time_col: str, value_col: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        if time_col not in df.columns or value_col not in df.columns:
            raise ValueError(f"文件列缺失: 需要[{time_col}, {value_col}]，实际为 {list(df.columns)}; 文件: {file_path}")
        df = df[[time_col, value_col]].copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df.sort_values(time_col, inplace=True)
        df.rename(columns={time_col: "Time", value_col: "Value"}, inplace=True)
        df = df[pd.notnull(df["Value"])]
        return df

    def plot(self, output_path: Path, show: bool = False, lag_minutes: int = 0) -> None:
        # (CDA列名, OMNI列名, 轴标签)
        param_pairs: List[Tuple[str, str, str]] = [
            ("BGSEc_0", "Bx", "Bx (GSE) [nT]"),
            ("BGSEc_1", "By_GSE", "By (GSE) [nT]"),
            ("BGSEc_2", "Bz_GSE", "Bz (GSE) [nT]"),
        ]
        series_map: Dict[str, Dict[str, pd.DataFrame]] = {}
        for cda_col, omni_col, _ in param_pairs:
            cda_file = self._find_file_by_suffix(self.data_dir, cda_col)
            omni_file = self._find_file_by_suffix(self.data_dir, omni_col)
            if cda_file is None:
                raise FileNotFoundError(f"未找到CDAWeb数据文件 (*_{cda_col}.csv) 于 {self.data_dir}")
            if omni_file is None:
                raise FileNotFoundError(f"未找到OMNI数据文件 (*_{omni_col}.csv) 于 {self.data_dir}")
            cda_df = self._load_series(cda_file, "Time", cda_col)
            omni_df = self._load_series(omni_file, "Time", omni_col)
            series_map[omni_col] = {"CDAWeb": cda_df, "OMNI": omni_df}

        # 标题可选：根据 OMNI Bx 文件名日期推断，这里保持简洁
        self._plotter.plot_from_series(
            series_map=series_map,
            param_pairs=param_pairs,
            output_path=output_path,
            title=None,
            show=show,
            lag_minutes=lag_minutes,
        )


class SourceComparatorUsingPkg:
    """
    通过 space_weather_data 包获取 OMNI 与 CDA 数据并比较绘图（Bx/By_GSE/Bz_GSE）。
    """
    def __init__(self, pkg_dir: Path, dataset: str = "AC_H0_MFI", cdaweb_datatype: Optional[str] = None,
                 omni_resolution: str = "1min"):
        self.pkg_dir = Path(pkg_dir)
        self.dataset = dataset
        self.cdaweb_datatype = cdaweb_datatype
        self.omni_resolution = omni_resolution
        _set_times_font()
        self._plotter = ComparisonPlotter()

    def _add_pkg_path(self):
        pkg_dir = self.pkg_dir.resolve()
        if str(pkg_dir) not in sys.path:
            sys.path.insert(0, str(pkg_dir))

    def _is_single_day(self, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> bool:
        try:
            return pd.to_datetime(start_dt).date() == pd.to_datetime(end_dt).date()
        except Exception:
            return False

    def _find_local_omni_file(self, day: pd.Timestamp) -> Optional[Path]:
        """
        寻找 OMNI 1min 单日 CDF：
        - OMNI 的 1min 文件按“月”归档，文件名用该月第一天 YYYYMM01
        - 先在 time_delay_analyze 下找：omni_data/hro_1min/YYYY/omni_hro_1min_YYYYMM01_v*.cdf
        - 再在 space_weather_data 下找：同结构
        """
        day = pd.to_datetime(day)
        yyyy = day.strftime("%Y")
        ymd_first = day.strftime("%Y%m") + "01"
        patterns = [
            f"omni_data/hro_1min/{yyyy}/omni_hro_1min_{ymd_first}_v*.cdf",
        ]
        base_dirs: List[Path] = [
            Path(__file__).resolve().parent,           # time_delay_analyze/...
            self.pkg_dir.resolve(),                    # space_weather_data/...
        ]
        for base in base_dirs:
            for pat in patterns:
                candidates = sorted(base.glob(pat))
                if candidates:
                    return candidates[-1]
        return None

    def _find_local_cda_file(self, day: pd.Timestamp) -> Optional[Path]:
        """
        寻找 CDA 单日 CDF，按 dataset 推断路径与文件名前缀：
        - AC_H0_MFI: ace_data/mag/level_2_cdaweb/mfi_h0/YYYY/ac_h0_mfi_YYYYMMDD_v*.cdf
        - AC_H3_MFI: ace_data/mag/level_2_cdaweb/mfi_h3/YYYY/ac_h3_mfi_YYYYMMDD_v*.cdf
        - WI_H0_MFI: wind_data/mfi/mfi_h0/YYYY/wi_h0_mfi_YYYYMMDD_v*.cdf
        """
        dataset = (self.dataset or "").upper()
        yyyy = pd.to_datetime(day).strftime("%Y")
        ymd = pd.to_datetime(day).strftime("%Y%m%d")
        patterns: List[str] = []
        if dataset == "AC_H0_MFI":
            patterns.append(f"ace_data/mag/level_2_cdaweb/mfi_h0/{yyyy}/ac_h0_mfi_{ymd}_v*.cdf")
        elif dataset == "AC_H3_MFI":
            patterns.append(f"ace_data/mag/level_2_cdaweb/mfi_h3/{yyyy}/ac_h3_mfi_{ymd}_v*.cdf")
        elif dataset == "WI_H0_MFI":
            patterns.append(f"wind_data/mfi/mfi_h0/{yyyy}/wi_h0_mfi_{ymd}_v*.cdf")
        else:
            # 未知数据集：不做本地查找
            return None
        base_dirs: List[Path] = [
            Path(__file__).resolve().parent,   # time_delay_analyze/...
            self.pkg_dir.resolve(),            # space_weather_data/...
        ]
        for base in base_dirs:
            for pat in patterns:
                candidates = sorted(base.glob(pat))
                if candidates:
                    return candidates[-1]
        return None

    def _load_omni_from_local(self, local_file: Path, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
        """
        使用本地 OMNI CDF 文件加载到 DataFrame，列名标准化为 ['Time','Bx','By_GSE','Bz_GSE']（若存在）。
        """
        self._add_pkg_path()
        from data_fetcher import DataFetcher  # 复用内部 DataFrame 构建逻辑
        try:
            import pyspedas
            from pytplot import del_data
        except Exception as e:
            print(f"[WARN] 本地加载 OMNI 失败（缺少依赖）：{e}")
            return pd.DataFrame()
        del_data('*')
        try:
            # 使用 cdf_to_tplot 直接从本地 CDF 生成 tplot 变量，避免远端索引
            loaded_vars = pyspedas.cdf_to_tplot([str(local_file)])
            fetcher = DataFetcher(resolution=self.omni_resolution)
            df = fetcher._process_pyspedas_data(loaded_vars, standardize_omni=True)
            if df.empty:
                return df
            # 仅保留需要列（加入速度/密度列，如存在）
            keep_omni = ["Time", "Bx", "By_GSE", "Bz_GSE", "Vsw", "Vx", "nsw"]
            df = df[[c for c in keep_omni if c in df.columns]].copy()
            # 按请求时间段截取（稳妥起见）
            t0 = pd.to_datetime(start_dt)
            t1 = pd.to_datetime(end_dt)
            if "Time" in df.columns:
                df = df[(df["Time"] >= t0) & (df["Time"] <= t1)]
            return df.reset_index(drop=True)
        except Exception as e:
            print(f"[WARN] 本地 OMNI 解析失败：{e}")
            return pd.DataFrame()

    def _load_cda_from_local(self, local_file: Path, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
        """
        使用本地 CDA CDF 文件加载到 DataFrame，输出列包含 ['Time','BX_GSE','BY_GSE','BZ_GSE']（若数据存在）。
        同时兼容 BGSE / BGSEc 分量命名。
        """
        self._add_pkg_path()
        from data_fetcher import DataFetcher
        try:
            import pyspedas
            from pytplot import del_data
        except Exception as e:
            print(f"[WARN] 本地加载 CDA 失败（缺少依赖）：{e}")
            return pd.DataFrame()
        del_data('*')
        try:
            # 直接从本地 CDF 读入，避免远端索引
            loaded_vars = pyspedas.cdf_to_tplot([str(local_file)])
            fetcher = DataFetcher(resolution=self.omni_resolution)
            df_raw = fetcher._process_pyspedas_data(loaded_vars)
            if df_raw.empty:
                return df_raw
            # 后处理：从 BGSE/BGSEc 组件中构建标准列
            out = pd.DataFrame()
            out["Time"] = df_raw["Time"]
            for idx, name in enumerate(["BX_GSE", "BY_GSE", "BZ_GSE"]):
                for base in ["BGSE", "BGSEc"]:
                    col = f"{base}_{idx}"
                    if col in df_raw.columns:
                        out[name] = df_raw[col]
                        break
            # 截取请求时间段
            t0 = pd.to_datetime(start_dt)
            t1 = pd.to_datetime(end_dt)
            out = out[(out["Time"] >= t0) & (out["Time"] <= t1)].reset_index(drop=True)
            # 若三列均缺失，返回空
            any_comp = any(c in out.columns for c in ["BX_GSE", "BY_GSE", "BZ_GSE"])
            if not any_comp:
                return pd.DataFrame()
            return out
        except Exception as e:
            print(f"[WARN] 本地 CDA 解析失败：{e}")
            return pd.DataFrame()

    def fetch(self, time_str: str, overwrite: bool = False, speed_only: bool = False) -> Dict[str, pd.DataFrame]:
        self._add_pkg_path()
        from time_parser import TimeParser
        from data_fetcher import DataFetcher

        parser = TimeParser()
        start_dt, end_dt = parser.parse(time_str)
        fetcher = DataFetcher(resolution=self.omni_resolution)

        # 单日且未指定覆盖时，优先尝试本地 CDF；两来源独立判断
        single_day = self._is_single_day(start_dt, end_dt)
        df_omni: pd.DataFrame
        df_cda: pd.DataFrame

        # OMNI
        df_omni = pd.DataFrame()
        if single_day and not overwrite:
            local_omni = self._find_local_omni_file(start_dt)
            if local_omni and local_omni.exists():
                print(f"[LOCAL] 使用本地 OMNI CDF: {local_omni}")
                df_omni = self._load_omni_from_local(local_omni, start_dt, end_dt)
        if df_omni.empty:
            # 远端获取
            df_omni = fetcher.fetch_omni(start_dt, end_dt)
            keep_omni = ["Time", "Bx", "By_GSE", "Bz_GSE", "Vsw", "Vx", "nsw"]
            df_omni = df_omni[[c for c in keep_omni if c in df_omni.columns]].copy()

        # CDA 参数：
        # - speed_only 模式：仅请求速度列（SWE: Vp；若非 SWE，尝试通用速度向量 V_* 或仅保留空）
        # - 否则：磁场三分量 +（若为 SWE）附加 Vp
        cda_params = ["BX_GSE", "BY_GSE", "BZ_GSE"]
        ds_upper = (self.dataset or "").upper()
        if speed_only:
            if "SWE" in ds_upper:
                cda_params = ["Vp"]
            else:
                # 非 SWE 速度数据集可扩展；默认尝试常见速度向量名（可能为空）
                cda_params = ["Vp", "V_GSE", "V_GSM", "V_RTN"]
        else:
            try:
                if "SWE" in ds_upper:
                    cda_params = cda_params + ["Vp"]
            except Exception:
                pass
        df_cda = pd.DataFrame()
        if single_day and not overwrite:
            local_cda = self._find_local_cda_file(start_dt)
            if local_cda and local_cda.exists():
                print(f"[LOCAL] 使用本地 CDA CDF: {local_cda}")
                df_cda = self._load_cda_from_local(local_cda, start_dt, end_dt)
        if df_cda.empty:
            df_cda = fetcher.fetch_cdaweb(self.dataset, start_dt, end_dt, cda_params, datatype=self.cdaweb_datatype)
            df_cda = df_cda[[c for c in ["Time"] + cda_params if c in df_cda.columns]].copy()
        return {"omni": df_omni, "cda": df_cda}

    def plot(self, time_str: str, output: Path, show: bool = False, lag_minutes: int = 0, overwrite: bool = False) -> None:
        data = self.fetch(time_str, overwrite=overwrite)
        omni = data["omni"]
        cda = data["cda"]

        components = [
            ("Bx", "Bx (GSE) [nT]", "BX_GSE"),
            ("By_GSE", "By (GSE) [nT]", "BY_GSE"),
            ("Bz_GSE", "Bz (GSE) [nT]", "BZ_GSE"),
        ]

        self._plotter.plot_from_dataframes(
            omni=omni,
            cda=cda,
            components=components,
            output=output,
            title_prefix="CDAWeb vs OMNI Magnetic Field (GSE)",
            show=show,
            lag_minutes=lag_minutes,
        )


class TimeDelayAnalyzer:
    """
    计算两来源之间的时延（基于滑动窗口MSE），同时完成：
    - 统一到1分钟分辨率
    - 对两来源做时间插值（time-based）
    """
    def __init__(self):
        _set_times_font()

    @staticmethod
    def _prepare_1min_series(omni_df: pd.DataFrame, cda_df: pd.DataFrame,
                             omni_col: str, cda_col: str) -> Tuple[pd.Series, pd.Series]:
        # 选择列并保证时间为datetime（不先 dropna，以原始覆盖求交集）
        df1 = omni_df[['Time', omni_col]].copy()
        df2 = cda_df[['Time', cda_col]].copy()
        df1['Time'] = pd.to_datetime(df1['Time'])
        df2['Time'] = pd.to_datetime(df2['Time'])
        # 求原始时间覆盖的交集范围
        start = max(df1['Time'].min(), df2['Time'].min())
        end = min(df1['Time'].max(), df2['Time'].max())
        if start >= end:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        # 构造统一1分钟索引（取整到分钟）
        start = start.floor('min')
        end = end.ceil('min')
        idx = pd.date_range(start=start, end=end, freq='1min')
        # 转数值并设为索引
        s1 = pd.to_numeric(df1.set_index('Time')[omni_col], errors='coerce').sort_index()
        s2 = pd.to_numeric(df2.set_index('Time')[cda_col], errors='coerce').sort_index()
        # 先正规化到 1 分钟分辨率（平均或聚合）
        s1r = s1.resample('1min').mean()
        s2r = s2.resample('1min').mean()
        # 对齐到统一索引并插值
        s1i = s1r.reindex(idx).interpolate(method='time', limit_direction='both').ffill().bfill()
        s2i = s2r.reindex(idx).interpolate(method='time', limit_direction='both').ffill().bfill()
        return s1i, s2i

    @staticmethod
    def _sliding_mse(s_ref: pd.Series, s_cmp: pd.Series, lag_range: range) -> pd.DataFrame:
        """
        对 s_cmp 进行整分钟平移（lag>0 表示 s_cmp 向后移，意味着 s_cmp 比 s_ref 更晚到达），
        计算与 s_ref 的MSE。
        """
        results = []
        for lag in lag_range:
            shifted = s_cmp.shift(lag)
            diff = s_ref - shifted
            mask = diff.notna()
            if mask.any():
                mse = float((diff[mask]**2).mean())
                n = int(mask.sum())
            else:
                mse = float('nan')
                n = 0
            results.append((lag, mse, n))
        df = pd.DataFrame(results, columns=['lag_min', 'mse', 'n'])
        return df

    def analyze_component(self, omni_df: pd.DataFrame, cda_df: pd.DataFrame,
                          omni_col: str, cda_col: str,
                          lag_min: int = -120, lag_max: int = 120) -> Dict[str, object]:
        s1, s2 = self._prepare_1min_series(omni_df, cda_df, omni_col, cda_col)
        if s1.empty or s2.empty:
            return {'component': omni_col, 'mse_curve': pd.DataFrame(), 'best_lag': None, 'best_mse': None}
        curve = self._sliding_mse(s1, s2, range(lag_min, lag_max + 1))
        if curve['mse'].notna().any():
            idx = curve['mse'].idxmin()
            best_lag = int(curve.loc[idx, 'lag_min'])
            best_mse = float(curve.loc[idx, 'mse'])
        else:
            best_lag = None
            best_mse = None
        return {'component': omni_col, 'mse_curve': curve, 'best_lag': best_lag, 'best_mse': best_mse}

    def analyze_all(self, omni_df: pd.DataFrame, cda_df: pd.DataFrame,
                    components: List[Tuple[str, str]]) -> Dict[str, Dict[str, object]]:
        """
        components: 列表 [(omni_col, cda_col), ...]
        """
        out: Dict[str, Dict[str, object]] = {}
        for omni_col, cda_col in components:
            out[omni_col] = self.analyze_component(omni_df, cda_df, omni_col, cda_col)
        return out

    def plot_mse_curves(self, results: Dict[str, Dict[str, object]], output: Path, show: bool = False) -> None:
        comps = ['Bx', 'By_GSE', 'Bz_GSE']
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)
        for ax, comp in zip(axes, comps):
            res = results.get(comp, {})
            curve: pd.DataFrame = res.get('mse_curve', pd.DataFrame())
            if not curve.empty:
                ax.plot(curve['lag_min'], curve['mse'], color='tab:purple', linewidth=1.2)
                if res.get('best_lag') is not None:
                    ax.axvline(res['best_lag'], color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.set_ylabel(f"{comp} MSE")
            ax.grid(True, linestyle='--', alpha=0.3)
        axes[-1].set_xlabel("Lag (minutes)  [positive: CDA arrives later]")
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)


