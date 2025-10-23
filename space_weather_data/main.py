"""Space weather data acquisition tool - CLI"""

import argparse
import sys
from datetime import datetime
from typing import List, Optional
import pandas as pd

from config import (
    PARAMETER_MAPPING, 
    DEFAULT_PARAMETERS, 
    DEFAULT_GOES_PARAMETERS,
    DATA_SOURCES,
    DATA_DIR
)
from time_parser import TimeParser
from data_fetcher import DataFetcher
from param_mapper import ParameterMapper
from data_manager import DataManager
from plotter import SpaceWeatherPlotter


class SpaceWeatherDataTool:
    """Main tool class"""
    
    def __init__(self):
        self.time_parser = TimeParser()
        self.param_mapper = ParameterMapper(PARAMETER_MAPPING)
        self.data_manager = DataManager(data_dir=DATA_DIR)
        self.plotter = SpaceWeatherPlotter()
    
    def run(self, time_input: str, parameters: List[str] = None,
            source: str = "omniweb", resolution: str = "hourly",
            output_filename: str = None, overwrite: bool = False,
            plot: bool = False, plot_file: str = None, **kwargs):
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
        
        # Parameter validation
        params_to_fetch = []
        if parameters is None:
            if source == 'omniweb':
                params_to_fetch = self.param_mapper.map(DEFAULT_PARAMETERS)
            elif source == 'cdaweb':
                print("\nError: --parameters are required for cdaweb source.")
                return False
            elif source == 'goes':
                instrument = kwargs.get('instrument')
                if instrument in DEFAULT_GOES_PARAMETERS:
                    params_to_fetch = self.param_mapper.map(DEFAULT_GOES_PARAMETERS[instrument])
                else:
                    print(f"\nError: No default parameters defined for GOES instrument '{instrument}'.")
                    return False
            elif source == 'indices':
                # Default to F107 and SSN
                params_to_fetch = ['F107', 'SSN']
            elif source == 'vtec':
                params_to_fetch = ['VTEC']
        else:
            params_to_fetch = self.param_mapper.map(parameters)

        if not params_to_fetch:
            print(f"\nError: No valid parameters specified for source '{source}'.")
            return False
        
        print(f"Parameters: {', '.join(params_to_fetch)}")
        print(f"Source: {DATA_SOURCES[source]['name']}")
        
        # Fetch data
        df = None
        try:
            fetcher = DataFetcher(resolution=resolution)

            if source == 'omniweb':
                df = fetcher.fetch_omni(start_dt, end_dt)
                # Post-filter for OMNI data, as pyspedas fetches all variables
                if df is not None and not df.empty:
                    columns_to_keep = ['Time']
                    for param in params_to_fetch:
                        if param in df.columns:
                            columns_to_keep.append(param)
                        else:
                            print(f"[INFO] Parameter '{param}' not found in OMNI data.")
                    df = df[columns_to_keep]

            elif source == 'cdaweb':
                dataset = kwargs.get('dataset')
                if not dataset:
                    print("\nError: --dataset is required for cdaweb source.")
                    return False
                # Parameters are passed directly to the fetcher for CDAWeb
                df = fetcher.fetch_cdaweb(dataset, start_dt, end_dt, params_to_fetch)
            
            elif source == 'goes':
                probe = kwargs.get('probe')
                instrument = kwargs.get('instrument')
                datatype = kwargs.get('datatype')
                if not probe or not instrument:
                    print("\nError: --probe and --instrument are required for goes source.")
                    return False
                df = fetcher.fetch_goes(
                    probe, start_dt, end_dt, instrument, datatype or '1min', 
                    requested_params=params_to_fetch
                )

            elif source == 'indices':
                # Fetch selected indices and outer-merge on Time
                frames = []
                wanted = set([p.upper() for p in params_to_fetch])
                if 'F107' in wanted:
                    f_df = fetcher.fetch_f107(start_dt, end_dt)
                    if f_df is not None and not f_df.empty:
                        frames.append(f_df)
                if 'SSN' in wanted:
                    s_df = fetcher.fetch_ssn(start_dt, end_dt)
                    if s_df is not None and not s_df.empty:
                        frames.append(s_df)
                if frames:
                    df = frames[0]
                    for x in frames[1:]:
                        df = pd.merge(df, x, on='Time', how='outer')
                    # Keep only requested columns
                    keep_cols = ['Time'] + [p for p in ['F107','SSN'] if p in wanted]
                    df = df[keep_cols].sort_values('Time')
                else:
                    df = pd.DataFrame()

            elif source == 'vtec':
                df = fetcher.fetch_vtec(start_dt, end_dt)


            if df is None or df.empty:
                print("\nError: No data retrieved")
                return False
            
            # --- Save the new data: one file per parameter ---
            self._save_data(df, start_dt, end_dt, source, resolution, params_to_fetch, overwrite, **kwargs)

        except Exception as e:
            print(f"\nError: {e}")
            return False
        
        if plot:
            try:
                print(f"\nGenerating plot...")
                self.plotter.plot(df, params_to_fetch, output_file=plot_file, show=True)
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
        print("Available OMNI Parameters")
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
        
        print(f"\nDefault OMNI: {', '.join(DEFAULT_PARAMETERS)}")
        print("\nFor CDAWeb, parameters are dataset-specific. Please consult CDAWeb documentation.")

    def _save_data(self, df: pd.DataFrame, start_dt: datetime, end_dt: datetime, 
                   source: str, resolution: str, params_to_fetch: List[str], 
                   overwrite: bool, **kwargs):
        """Helper function to save dataframe to CSV files, one per parameter."""
        if df is None or df.empty:
            return

        print("\nSaving data to separate files...")
        saved_files = []
        # Use a copy of kwargs to avoid modifying the original dict
        save_kwargs = kwargs.copy()
        save_kwargs['source'] = source
        save_kwargs['resolution'] = resolution

        for param in params_to_fetch:
            if param in df.columns:
                param_df = df[['Time', param]].dropna()
                
                if param_df.empty:
                    print(f"[INFO] No data for '{param}', skipping.")
                    continue

                param_filename = self.data_manager.get_filename(
                    start_dt, end_dt, param=param, **save_kwargs
                )
                
                result = self.data_manager.save_to_csv(param_df, param_filename, overwrite=overwrite)
                if result:
                    saved_files.append(param_filename)
        
        if not saved_files:
            print("[ERROR] No data could be saved.")
            return
        
        print(f"\n[OK] Data saved for {len(saved_files)} parameter(s).")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Space Weather Data Acquisition Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OMNI (default source)
  python main.py 2023-06 -p Bz Vsw       # Download specific parameters for a month
  python main.py 2023-06-15 --plot       # Download default OMNI parameters and plot

  # CDAWeb
  python main.py 2010-01-01 -s cdaweb --dataset WI_H0_MFI -p BZ_GSE   # Download a specific variable
  python main.py 2010-01-01 -s cdaweb --dataset WI_H0_MFI -p BX BY BZ --plot # Plot multiple variables
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
        help=f'Parameters/variables to fetch.'
    )
    
    parser.add_argument(
        '-s', '--source',
        default='omniweb',
        choices=list(DATA_SOURCES.keys()),
        help='Data source (default: omniweb)'
    )
    
    parser.add_argument(
        '--dataset',
        help='CDAWeb dataset ID (e.g., WI_H0_MFI, AC_H0_MFI)'
    )
    parser.add_argument(
        '--probe',
        help='GOES satellite probe number (e.g., 15, 16)'
    )
    parser.add_argument(
        '--instrument',
        help='GOES instrument (mag, particles, xrs)'
    )
    parser.add_argument(
        '--datatype',
        help='GOES data type (e.g., 1min, ep8)'
    )
    
    parser.add_argument(
        '-r', '--resolution',
        default='hourly',
        choices=['1min', '5min', 'hourly'],
        help='Time resolution for OMNI (default: hourly)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output filename (not used when saving one file per parameter)'
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
        help='Show available OMNI parameters'
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
        plot_file=args.plot_file,
        dataset=args.dataset,
        probe=args.probe,
        instrument=args.instrument,
        datatype=args.datatype
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

