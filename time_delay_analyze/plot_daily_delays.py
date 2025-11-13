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
import glob


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


def print_negative_lags(df: pd.DataFrame, columns: List[str]) -> None:
	"""Print dates with negative lag (per column)."""
	try:
		has_any = False
		for col in columns:
			if col not in df.columns:
				continue
			vals = pd.to_numeric(df[col], errors='coerce')
			mask = vals < 0
			if mask.any():
				has_any = True
				dates = pd.to_datetime(df.loc[mask, 'Date'])
				print(f"[INFO] Column {col} has negative lag on {int(mask.sum())} day(s):")
				for d, v in zip(dates.dt.strftime('%Y-%m-%d'), vals[mask]):
					try:
						print(f"  - {d}: {v}")
					except Exception:
						print(f"  - {d}")
		if not has_any:
			print("[INFO] No negative-lag dates found.")
	except Exception as e:
		print(f"[WARN] Failed to check negative lags: {e}", file=sys.stderr)


def clean_negative_values(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
	"""Replace negative values with NaN so they are excluded from plots/statistics."""
	try:
		df2 = df.copy()
		for col in columns:
			if col not in df2.columns:
				continue
			vals = pd.to_numeric(df2[col], errors='coerce')
			vals = vals.mask(vals < 0, np.nan)
			df2[col] = vals
		return df2
	except Exception:
		return df


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Plot daily best lags and overall histograms from CSV(s)")
	parser.add_argument("--csv", type=str, required=False, help="CSV file path, or a pattern (e.g., lag_2020.csv or daily_best_lags.csv)")
	parser.add_argument("--csv-glob", type=str, required=False, help="Glob pattern (e.g., lag_*.csv); can be used with --csv")
	parser.add_argument("--out-line", type=Path, default=Path(__file__).resolve().parent / "daily_best_lags.png",
						help="Output: daily lag line plot")
	parser.add_argument("--out-hist", type=Path, default=Path(__file__).resolve().parent / "daily_lag_hist.png",
						help="Output: overall lag histogram")
	parser.add_argument("--out-yearly", type=Path, default=Path(__file__).resolve().parent / "yearly_lag_stats.png",
						help="Output: yearly mean/variance line plot")
	parser.add_argument("--show", action="store_true", help="Show figure windows")
	parser.add_argument("--overlay-theory", action="store_true", help="Overlay theoretical lag (column theory_lag_min) on daily line plot")
	args = parser.parse_args(argv)

	set_times_font()

	# 收集文件列表（支持单文件与通配符）
	files: List[str] = []
	if args.csv:
		# 若包含通配符，expand；否则按单文件处理
		if any(ch in args.csv for ch in ["*", "?", "[", "]"]):
			files.extend(sorted(glob.glob(args.csv)))
		else:
			files.append(args.csv)
	if args.csv_glob:
		files.extend(sorted(glob.glob(args.csv_glob)))
	# 去重
	files = sorted(list(dict.fromkeys(files)))

	if not files:
		print("[ERROR] No CSVs provided or matched. Use --csv or --csv-glob.", file=sys.stderr)
		return 2

	frames: List[pd.DataFrame] = []
	is_multi = len(files) > 1
	for f in files:
		try:
			frames.append(pd.read_csv(f))
		except Exception as e:
			print(f"[警告] 读取失败，跳过: {f} ({e})", file=sys.stderr)

	if not frames:
		print("[ERROR] Failed to read any CSV.", file=sys.stderr)
		return 2

	try:
		df = pd.concat(frames, axis=0, ignore_index=True)
	except Exception as e:
		print(f"[ERROR] Failed to merge CSVs: {e}", file=sys.stderr)
		return 2

	if 'Date' not in df.columns:
		print("[ERROR] CSV missing 'Date' column.", file=sys.stderr)
		return 2

	# 打印 lag<0 的日期
	target_cols = ["Bx_lag", "By_GSE_lag", "Bz_GSE_lag"]
	print_negative_lags(df, target_cols)
	# 清洗负值（置为 NaN，不参与后续绘图与统计）
	df = clean_negative_values(df, target_cols)
	# 解析日期
	x_all = pd.to_datetime(df['Date'])

    # 折线图（3子图），横轴紧
	try:
		fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharex=True)
		x = x_all
		series = [("Bx_lag", "Bx delay (min)"), ("By_GSE_lag", "By (GSE) delay (min)"), ("Bz_GSE_lag", "Bz (GSE) delay (min)")]
		for ax, (col, ylabel) in zip(axes, series):
			y = pd.to_numeric(df.get(col, pd.Series([])), errors='coerce')
			# 多文件模式：仅折线，无点标记；单文件保留小圆点
			mk = None if is_multi else 'o'
			ms = 0.0 if is_multi else 3.0
			line_actual, = ax.plot(x, y, marker=mk, markersize=ms, linewidth=1.2, color='k', alpha=1.0, label='Actual')
			legend_handles = [line_actual]
			# 叠加理论延时（若开启并存在列）
			if args.overlay_theory and ('theory_lag_min' in df.columns):
				y_th = pd.to_numeric(df.get('theory_lag_min', pd.Series([])), errors='coerce')
				line_theory, = ax.plot(x, y_th, linestyle=':', linewidth=0.9, color='k', alpha=1.0, label='Theoretical')
				legend_handles.append(line_theory)
				# 比值折线（右侧 y 轴）：ratio = theory / actual
				line_ratio = None
				try:
					a = y.to_numpy(dtype=float)
					b = y_th.to_numpy(dtype=float)
					ratio = np.full_like(a, np.nan, dtype=float)
					mask = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 1e-12)
					ratio[mask] = b[mask] / a[mask]
					ax_r = ax.twinx()
					line_ratio, = ax_r.plot(x, ratio, linestyle='--', linewidth=0.9, color='0.4', alpha=1.0, label='Ratio (theory/actual)')
					ax_r.tick_params(axis='y', colors='0.2')
					# Fix ratio y-axis range
					ax_r.set_ylim(0.3, 5.0)
					# 右下角标注比值的均值与方差
					if np.isfinite(ratio).any():
						r_mean = float(np.nanmean(ratio))
						r_var = float(np.nanvar(ratio))
						ax.text(0.98, 0.02, f"Ratio mean={r_mean:.2f}\nVar={r_var:.2f}", transform=ax.transAxes,
							ha='right', va='bottom', color='k')
				except Exception:
					line_ratio = None
			ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=1.0)
			ax.set_ylabel(ylabel, color='k')
			ax.grid(True, linestyle='--', color='0.5', alpha=0.5)
			# 统一 Y 轴范围为 -10 ~ 120
			ax.set_ylim(-10.0, 120.0)
			# 图例（右上角，包含比值曲线）
			if args.overlay_theory and ('theory_lag_min' in df.columns) and ('line_ratio' in locals()) and (line_ratio is not None):
				ax.legend(handles=legend_handles + [line_ratio], loc='upper right', frameon=False)
			else:
				ax.legend(handles=legend_handles, loc='upper right', frameon=False)
		axes[-1].set_xlabel("Date (UTC)", color='k')
		tighten_x_axis(axes, x, ["Bx_lag", "By_GSE_lag", "Bz_GSE_lag"], df)
		fig.tight_layout()
		args.out_line.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(args.out_line, dpi=150)
		if args.show:
			plt.show()
		else:
			plt.close(fig)
		print(f"[OK] Saved daily lag line plot: {args.out_line}")
	except Exception as e:
		print(f"[WARN] Failed to draw daily line plot: {e}", file=sys.stderr)

    # 总体直方图（3子图）
	try:
		fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), sharex=False)
		series = [("Bx_lag", "Bx delay (min)"), ("By_GSE_lag", "By (GSE) delay (min)"), ("Bz_GSE_lag", "Bz (GSE) delay (min)")]
		for ax, (col, xlabel) in zip(axes, series):
			vals = pd.to_numeric(df.get(col, pd.Series([])), errors='coerce').dropna()
			if len(vals) > 0:
				# 固定横轴范围与 bin
				bins = np.arange(-120.0, 120.0 + 1.0, 1.0)
				ax.hist(vals, bins=bins, histtype='step', color='k', linewidth=1.0)
			# 0 轴线（黑白）
			ax.axvline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=1.0)
			ax.set_xlim(-120.0, 120.0)
			ax.set_xlabel(xlabel, color='k')
			ax.set_ylabel("Count", color='k')
			ax.grid(True, linestyle='--', color='0.5', alpha=0.5)
			# 方差标注（右上角）
			if len(vals) > 0:
				mean_val = float(np.nanmean(vals))
				var_val = float(np.nanvar(vals))
				ax.text(0.98, 0.95, f"Mean={mean_val:.2f}\nVar={var_val:.2f}", transform=ax.transAxes, ha='right', va='top', color='k')
		fig.tight_layout()
		args.out_hist.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(args.out_hist, dpi=150)
		if args.show:
			plt.show()
		else:
			plt.close(fig)
		print(f"[OK] Saved overall lag histogram: {args.out_hist}")
	except Exception as e:
		print(f"[WARN] Failed to draw histogram: {e}", file=sys.stderr)

	# 多年情况下：按年统计均值与方差并绘制（同一张图每列两条线）
	try:
		years = x_all.dt.year
		df_year = df.copy()
		df_year['__year__'] = years
		if df_year['__year__'].notna().any():
			fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharex=True)
			series = [("Bx_lag", "Bx yearly stats"), ("By_GSE_lag", "By (GSE) yearly stats"), ("Bz_GSE_lag", "Bz (GSE) yearly stats")]
			for ax, (col, ylabel) in zip(axes, series):
				if col not in df_year.columns:
					continue
				g = df_year.groupby('__year__')[col]
				years_sorted = sorted([int(y) for y in g.groups.keys() if pd.notnull(y)])
				if not years_sorted:
					continue
				mean_vals = [np.nanmean(pd.to_numeric(g.get_group(y), errors='coerce')) if y in g.groups else np.nan for y in years_sorted]
				var_vals = [np.nanvar(pd.to_numeric(g.get_group(y), errors='coerce')) if y in g.groups else np.nan for y in years_sorted]
				ax.plot(years_sorted, mean_vals, color='k', linestyle='-', marker='o', markersize=3.0, linewidth=1.2, label='Mean')
				ax.plot(years_sorted, var_vals, color='k', linestyle='--', marker='s', markersize=3.0, linewidth=1.0, label='Var')
				ax.set_ylabel(ylabel, color='k')
				ax.grid(True, linestyle='--', color='0.5', alpha=0.5)
				# 横轴紧：限定至最小/最大年份并去掉边距
				try:
					ax.set_xlim(min(years_sorted), max(years_sorted))
					ax.set_xmargin(0)
					ax.margins(x=0)
				except Exception:
					pass
				ax.legend(loc='upper right', frameon=False)
			axes[-1].set_xlabel("Year", color='k')
			fig.tight_layout()
			args.out_yearly.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(args.out_yearly, dpi=150)
			if args.show:
				plt.show()
			else:
				plt.close(fig)
			print(f"[OK] Saved yearly mean/variance line plot: {args.out_yearly}")
	except Exception as e:
		print(f"[WARN] Failed to draw yearly statistics plot: {e}", file=sys.stderr)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())


