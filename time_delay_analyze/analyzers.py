#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import DateFormatter, AutoDateLocator

import sys


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

    def fetch(self, time_str: str) -> Dict[str, pd.DataFrame]:
        self._add_pkg_path()
        from time_parser import TimeParser
        from data_fetcher import DataFetcher

        parser = TimeParser()
        start_dt, end_dt = parser.parse(time_str)
        fetcher = DataFetcher(resolution=self.omni_resolution)

        df_omni = fetcher.fetch_omni(start_dt, end_dt)
        keep_omni = ["Time", "Bx", "By_GSE", "Bz_GSE"]
        df_omni = df_omni[[c for c in keep_omni if c in df_omni.columns]].copy()

        cda_params = ["BX_GSE", "BY_GSE", "BZ_GSE"]
        df_cda = fetcher.fetch_cdaweb(self.dataset, start_dt, end_dt, cda_params, datatype=self.cdaweb_datatype)
        df_cda = df_cda[[c for c in ["Time"] + cda_params if c in df_cda.columns]].copy()
        return {"omni": df_omni, "cda": df_cda}

    def plot(self, time_str: str, output: Path, show: bool = False, lag_minutes: int = 0) -> None:
        data = self.fetch(time_str)
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


