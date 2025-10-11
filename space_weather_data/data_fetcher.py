"""Data fetching module - refactored to use pyspedas library"""

import pandas as pd
from datetime import datetime
from typing import List

try:
    # Import the top-level pyspedas library
    import pyspedas
    from pytplot import get_data, del_data
    # GOES netCDF files are now loaded using the core pytplot function
    from pytplot import netcdf_to_tplot as goes_netcdf_to_tplot
    PYSPEDAS_AVAILABLE = True
except ImportError as e:
    print(f"ImportError: {e}")
    PYSPEDAS_AVAILABLE = False

# Mapping from pyspedas tplot variable names to our project's standard names
PYSPEDAS_COLUMN_MAPPING = {
    'Epoch': 'Time', 'F': 'B', 'BX_GSE': 'Bx', 'BY_GSE': 'By_GSE', 'BZ_GSE': 'Bz_GSE',
    'BY_GSM': 'By', 'BZ_GSM': 'Bz', 'flow_speed': 'Vsw', 'V': 'Vsw', 'Vx': 'Vx',
    'Vy': 'Vy', 'Vz': 'Vz', 'proton_density': 'nsw', 'N': 'nsw', 'T': 'Tsw',
    'Pressure': 'Psw', 'E': 'Esw', 'beta': 'beta', 'Mach_num': 'Mach_num', 'Kp': 'Kp',
    'AE_INDEX': 'AE', 'AL_INDEX': 'AL', 'AU_INDEX': 'AU', 'SYM_H': 'SYM-H',
    'SYM_D': 'SYM-D', 'ASY_H': 'ASYM-H', 'ASY_D': 'ASYM-D', 'PC_N_INDEX': 'PC', 'DST': 'DST'
}

# Translates user-friendly standard names to dataset-specific pyspedas variables and components
CDAWEB_PARAM_MAPPING = {
    'WI_H0_MFI': {
        # user_param: (pyspedas_variable, component_index)
        'BX_GSE': ('BGSE', 0),
        'BY_GSE': ('BGSE', 1),
        'BZ_GSE': ('BGSE', 2),
    },
    'AC_H0_MFI': {
        'BX_GSE': ('BGSE', 0),
        'BY_GSE': ('BGSE', 1),
        'BZ_GSE': ('BGSE', 2),
    }
    # Future datasets can be added here
}

class DataFetcher:
    """
    Unified data fetcher for OMNI and specific CDAWeb datasets using the pyspedas library.
    Follows the mission-specific pyspedas loading pattern.
    """
    
    def __init__(self, resolution: str = "hourly"):
        if not PYSPEDAS_AVAILABLE:
            raise ImportError("pyspedas library not found or incomplete. Please run: pip install pyspedas")
        self.resolution = resolution
        # Map user-friendly dataset IDs to the actual pyspedas functions
        self.cdaweb_function_mapping = {
            'WI_H0_MFI': pyspedas.wind.mfi,
            'AC_H0_MFI': pyspedas.ace.mfi,      # Corrected from 'mag' to 'mfi'
            'AC_H1_SWE': pyspedas.ace.swe,      # Corrected from 'swepam' to 'swe'
            'MMS1_FGM_SRVY_L2': pyspedas.mms.fgm,
        }
        self.goes_function_mapping = {
            'mag': pyspedas.goes.mag,
            'particles': pyspedas.goes.eps,      # Corrected from 'particles' to 'eps'
            'xrs': pyspedas.goes.xrs
        }

    def _build_goes_pyspedas_name(self, simple_name: str, probe: str, instrument: str) -> str:
        """Constructs the full pyspedas variable name from a simple name."""
        if instrument == 'xrs':
            # Handle different naming conventions for GOES probes
            try:
                probe_num = int(probe)
            except (ValueError, TypeError):
                probe_num = 0  # Default if probe is not a number

            if probe_num > 15:
                # GOES-R series (16+) use a _flux suffix.
                return f"{simple_name}_flux"
            else:
                # Older GOES satellites (<=15) use A_AVG, B_AVG for x-ray flux
                if simple_name == 'xrsa':
                    return 'A_AVG'
                if simple_name == 'xrsb':
                    return 'B_AVG'
                # Fallback for other xrs params on old probes, though likely not present
                return simple_name
        
        # particles uses an 'eps' prefix inside the variable name
        if instrument == 'particles':
            return f"g{probe}_eps_{simple_name}"
            
        # mag is the default case
        return f"g{probe}_{simple_name}"

    def _process_pyspedas_data(self, loaded_vars: List[str], standardize_omni: bool = False) -> pd.DataFrame:
        """Helper function to convert tplot variables to a pandas DataFrame."""
        if not loaded_vars:
            print("[INFO] No data found for the specified criteria.")
            return pd.DataFrame()

        all_dfs = []
        for var in loaded_vars:
            var_data = get_data(var)
            if var_data is None: continue
            
            df_var = pd.DataFrame(index=pd.to_datetime(var_data.times, unit='s'), data=var_data.y)
            
            if len(df_var.columns) > 1:
                df_var.columns = [f"{var}_{i}" for i in range(len(df_var.columns))]
            else:
                df_var.columns = [var]
            all_dfs.append(df_var)

        if not all_dfs: return pd.DataFrame()

        df = pd.concat(all_dfs, axis=1)
        df = df.reset_index().rename(columns={'index': 'Time'})
        
        if standardize_omni:
            df = df.rename(columns=PYSPEDAS_COLUMN_MAPPING)
        
        return df
    
    def fetch_omni(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch OMNI data."""
        print(f"\nFetching OMNI data (via pyspedas)...")
        print(f"Resolution: {self.resolution}")
        del_data('*')
        time_range = [start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')]
        
        try:
            loaded_vars = pyspedas.omni.data(trange=time_range, datatype=self.resolution)
            df = self._process_pyspedas_data(loaded_vars, standardize_omni=True)
            print(f"\n[OK] Retrieved {len(df)} OMNI records via pyspedas")
            return df
        except Exception as e:
            print(f"Failed to download OMNI data using pyspedas: {e}")
            return pd.DataFrame()

    def fetch_cdaweb(self, dataset: str, start_dt: datetime, end_dt: datetime, parameters: List[str]) -> pd.DataFrame:
        """Fetch a specific CDAWeb dataset, translating standard parameter names."""
        print(f"\nFetching CDAWeb data (via pyspedas)...")
        print(f"Dataset: {dataset}")
        del_data('*')
        
        load_function = self.cdaweb_function_mapping.get(dataset)
        if not load_function:
            raise ValueError(f"Unsupported dataset '{dataset}'. Supported: {list(self.cdaweb_function_mapping.keys())}")

        # --- Parameter Translation Logic ---
        pyspedas_vars_to_load = set()
        component_params = {} # Maps user_param -> (pyspedas_var, index)
        dataset_map = CDAWEB_PARAM_MAPPING.get(dataset, {})

        for p in parameters:
            if p in dataset_map:
                pyspedas_var, index = dataset_map[p]
                pyspedas_vars_to_load.add(pyspedas_var)
                component_params[p] = (pyspedas_var, index)
            else:
                # If param is not in our map, assume it's a direct variable name
                pyspedas_vars_to_load.add(p)

        if not pyspedas_vars_to_load:
            print(f"[ERROR] Could not map requested parameters to any known variables for dataset '{dataset}'.")
            return pd.DataFrame()
        # --- End Translation Logic ---

        time_range = [start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')]

        try:
            loaded_vars = load_function(trange=time_range, varnames=list(pyspedas_vars_to_load))
            df_raw = self._process_pyspedas_data(loaded_vars)

            if df_raw.empty:
                print(f"\n[INFO] pyspedas loaded the dataset but found no data for the requested variables.")
                return df_raw

            # --- Post-processing: Build final DataFrame with user-requested names ---
            final_df = pd.DataFrame()
            final_df['Time'] = df_raw['Time']

            for user_param, (pyspedas_var, index) in component_params.items():
                pyspedas_col_name = f"{pyspedas_var}_{index}"
                if pyspedas_col_name in df_raw.columns:
                    final_df[user_param] = df_raw[pyspedas_col_name]
            
            # Include any parameters that were passed directly and not mapped
            for p in parameters:
                if p not in component_params and p in df_raw.columns:
                    final_df[p] = df_raw[p]
            # --- End Post-processing ---

            print(f"\n[OK] Retrieved {len(final_df)} CDAWeb records via pyspedas")
            return final_df
        except Exception as e:
            print(f"Failed to download CDAWeb data for dataset '{dataset}': {e}")
            return pd.DataFrame()

    def fetch_goes(self, probe: str, start_dt: datetime, end_dt: datetime, instrument: str, datatype: str, requested_params: List[str]) -> pd.DataFrame:
        """Fetch GOES data, handling parameter name mapping internally."""
        print(f"\nFetching GOES data (via pyspedas)...")
        print(f"Probe: {probe}, Instrument: {instrument}, Datatype: {datatype}")
        del_data('*')

        load_function = self.goes_function_mapping.get(instrument)
        if not load_function:
            raise ValueError(f"Unsupported GOES instrument '{instrument}'. Supported: {list(self.goes_function_mapping.keys())}")

        time_range = [start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')]

        try:
            # Step 1: Download the files only, don't attempt to load them with the buggy loader.
            # This returns a list of local file paths.
            files = load_function(
                trange=time_range, probe=probe, datatype=datatype, 
                downloadonly=True
            )

            if not files:
                print("[INFO] No GOES data files found for the specified time range.")
                return pd.DataFrame()

            # Step 2: Manually load the downloaded files using the low-level netcdf_to_tplot,
            # which allows us to correctly specify the variable names.
            pyspedas_vars_to_load = [
                self._build_goes_pyspedas_name(p, probe, instrument)
                for p in requested_params
            ]
            
            loaded_vars = goes_netcdf_to_tplot(files)
            
            df_raw = self._process_pyspedas_data(loaded_vars)

            if df_raw.empty:
                return df_raw
            
            # --- Filter and rename columns to match the simple names requested by the user ---
            final_df = pd.DataFrame()
            final_df['Time'] = df_raw['Time']
            
            for i, simple_param in enumerate(requested_params):
                # Use the same full pyspedas name that we requested
                pyspedas_name = pyspedas_vars_to_load[i]

                if pyspedas_name in df_raw.columns:
                    final_df[simple_param] = df_raw[pyspedas_name]
                else:
                    print(f"[INFO] Requested parameter '{simple_param}' (expected as '{pyspedas_name}') not found in GOES data.")

            # Drop Time column if it's the only one left (meaning no params were found)
            if len(final_df.columns) == 1 and 'Time' in final_df.columns:
                return pd.DataFrame()

            print(f"\n[OK] Retrieved {len(final_df)} GOES records via pyspedas")
            return final_df

        except Exception as e:
            print(f"Failed to download GOES data: {e}")
            return pd.DataFrame()
