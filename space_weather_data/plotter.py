"""Data plotting module"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from typing import List


class SpaceWeatherPlotter:
    """Space weather data plotter"""
    
    def __init__(self):
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        self.units = {
            'Bz': 'nT',
            'By': 'nT',
            'Bx': 'nT',
            'B': 'nT',
            'Vsw': 'km/s',
            'nsw': 'N/cm³',
            'Tsw': 'K',
            'Psw': 'nPa',
            'Esw': 'mV/m',
            'AE': 'nT',
            'AL': 'nT',
            'AU': 'nT',
            'SYM-H': 'nT',
            'DST': 'nT',
            'ASYM-H': 'nT',
            'PC': '',
        }

    
    def plot(self, df: pd.DataFrame, parameters: List[str], 
             output_file: str = None, show: bool = True):
        """Plot time series data"""
        if 'Time' not in df.columns:
            raise ValueError("DataFrame must contain 'Time' column")
        
        if not pd.api.types.is_datetime64_any_dtype(df['Time']):
            df['Time'] = pd.to_datetime(df['Time'])
        
        valid_params = [p for p in parameters if p in df.columns]
        if not valid_params:
            print("Warning: No valid parameters to plot")
            return
        
        n_params = len(valid_params)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params), 
                                 sharex=True, squeeze=False)
        axes = axes.flatten()
        
        for i, param in enumerate(valid_params):
            ax = axes[i]
            ax.plot(df['Time'], df[param], linewidth=1.5, color='#1f77b4')
            
            unit = self.units.get(param, '')
            title = f"{param} ({unit})" if unit else param
            ax.set_ylabel(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            if param in ['Bz', 'By', 'Bx', 'SYM-H', 'DST']:
                ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
            
            y_data = df[param].dropna()
            if len(y_data) > 0:
                y_min, y_max = y_data.min(), y_data.max()
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        
        axes[-1].set_xlabel('Time (UTC)', fontsize=11, fontweight='bold')
        
        for ax in axes:
            ax.set_xlim(df['Time'].min(), df['Time'].max())
        
        self._format_time_axis(axes[-1], df['Time'])
        
        time_range = f"{df['Time'].min().strftime('%Y-%m-%d')} to {df['Time'].max().strftime('%Y-%m-%d')}"
        fig.suptitle(f'Space Weather Data\n{time_range}', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"[OK] Image saved: {output_file}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def _format_time_axis(self, ax, time_series):
        """Format time axis display"""
        time_range = (time_series.max() - time_series.min()).total_seconds()
        
        if time_range < 86400 * 2:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(time_range/3600/10))))
        elif time_range < 86400 * 60:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(time_range/86400/10))))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, int(time_range/86400/30/10))))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def quick_plot(self, csv_file: str, parameters: List[str] = None):
        """Quick plot from CSV file"""
        df = pd.read_csv(csv_file)
        if parameters is None:
            parameters = [col for col in df.columns if col != 'Time']
        self.plot(df, parameters)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        plotter = SpaceWeatherPlotter()
        plotter.quick_plot(sys.argv[1])
    else:
        print("Usage: python plotter.py <csv_file>")

