"""Space weather data acquisition tool - CLI"""

import argparse
import sys
from datetime import datetime
from typing import List

from config import (
    PARAMETER_MAPPING, 
    DEFAULT_PARAMETERS, 
    DATA_SOURCES,
    DATA_DIR
)
from time_parser import TimeParser
from data_fetcher import OMNIWebFetcher
from data_manager import DataManager
from plotter import SpaceWeatherPlotter


class SpaceWeatherDataTool:
    """Main tool class"""
    
    def __init__(self):
        self.time_parser = TimeParser()
        self.data_manager = DataManager(data_dir=DATA_DIR)
        self.plotter = SpaceWeatherPlotter()
    
    def run(self, time_input: str, parameters: List[str] = None,
            source: str = "omniweb", resolution: str = "hourly",
            output_filename: str = None, overwrite: bool = False,
            plot: bool = False, plot_file: str = None):
        print("\n" + "="*60)
        print("Space Weather Data Acquisition")
        print("="*60)
        
        # Parse time
        try:
            start_dt, end_dt = self.time_parser.parse(time_input)
            print(f"\nTime range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        except ValueError as e:
            print(f"\nError: {e}")
            return False
        
        if parameters is None:
            parameters = DEFAULT_PARAMETERS
        
        valid_params = [p for p in parameters if p in PARAMETER_MAPPING]
        invalid_params = [p for p in parameters if p not in PARAMETER_MAPPING]
        
        if invalid_params:
            print(f"\nWarning: Invalid parameters ignored: {', '.join(invalid_params)}")
        
        if not valid_params:
            print(f"\nError: No valid parameters")
            return False
        
        print(f"Parameters: {', '.join(valid_params)}")
        print(f"Source: {DATA_SOURCES[source]['name']}")
        
        try:
            fetcher = OMNIWebFetcher(resolution=resolution)
            df = fetcher.fetch(start_dt, end_dt, valid_params)
            
            if df is None or df.empty:
                print("\nError: No data retrieved")
                return False
            
            print(f"\nData preview:")
            print(df.head())
            
        except Exception as e:
            print(f"\nError: {e}")
            return False
        
        if output_filename is None:
            output_filename = self.data_manager.get_filename(start_dt, end_dt, source, resolution)
        
        result = self.data_manager.save_to_csv(df, output_filename, overwrite=overwrite)
        
        if not result:
            print(f"\n[ERROR] Save failed")
            return False
        
        if plot:
            try:
                print(f"\nGenerating plot...")
                self.plotter.plot(df, valid_params, output_file=plot_file, show=True)
            except Exception as e:
                print(f"\nWarning: Plot failed - {e}")
        
        print(f"\n[OK] Done!")
        return True
    
    def list_data(self):
        """List saved data files"""
        print("\n" + "="*60)
        print("Local Data Files")
        print("="*60)
        print(self.data_manager.get_data_info())
    
    def show_available_parameters(self):
        """Show available parameters"""
        print("\n" + "="*60)
        print("Available Parameters")
        print("="*60)
        
        categories = {
            "Magnetic Field": ["B", "Bx", "By", "Bz"],
            "Solar Wind": ["Vsw", "nsw", "Tsw"],
            "Pressure & Energy": ["Psw", "Esw"],
            "Geomagnetic Indices": ["AE", "AL", "AU", "SYM-H", "SYM-D", "ASYM-H", "ASYM-D", "PC"],
        }
        
        for category, params in categories.items():
            print(f"\n{category}:")
            for param in params:
                if param in PARAMETER_MAPPING:
                    print(f"  - {param}")
        
        print(f"\nDefault: {', '.join(DEFAULT_PARAMETERS)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Space Weather Data Acquisition Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py 2023                    # Full year
  python main.py 2023-06 -p Bz Vsw nsw  # Specific month & parameters
  python main.py 2023-06-15 --plot       # Single day with plot
  python main.py --list                  # List local files
  python main.py --show-params           # Show available parameters
        """
    )
    
    parser.add_argument(
        'time', 
        nargs='?',
        help='Time input (e.g., 2023, 2023-06, 2023-06-15, 2020-2023)'
    )
    
    parser.add_argument(
        '-p', '--parameters',
        nargs='+',
        help=f'Parameters to fetch (default: {" ".join(DEFAULT_PARAMETERS[:3])}...)'
    )
    
    parser.add_argument(
        '-s', '--source',
        default='omniweb',
        help='Data source (default: omniweb)'
    )
    
    parser.add_argument(
        '-r', '--resolution',
        default='hourly',
        choices=['1min', '5min', 'hourly'],
        help='Time resolution (default: hourly)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output filename (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing file'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List local data files'
    )
    
    parser.add_argument(
        '--show-params',
        action='store_true',
        help='Show available parameters'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Plot data'
    )
    
    parser.add_argument(
        '--plot-file',
        help='Plot filename (auto-generated if not specified)'
    )
    
    args = parser.parse_args()
    
    tool = SpaceWeatherDataTool()
    
    if args.list:
        tool.list_data()
        return
    
    if args.show_params:
        tool.show_available_parameters()
        return
    
    if not args.time:
        parser.print_help()
        return
    
    success = tool.run(
        time_input=args.time,
        parameters=args.parameters,
        source=args.source,
        resolution=args.resolution,
        output_filename=args.output,
        overwrite=args.overwrite,
        plot=args.plot,
        plot_file=args.plot_file
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

