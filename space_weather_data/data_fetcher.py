"""Data fetching module - refactored to use pyspedas library

Adds lightweight helpers to fetch solar indices (F107, SSN) from
public NOAA/SWPC endpoints to complement OMNI/CDAWeb/GOES.
"""

import pandas as pd
import requests
import os
from pathlib import Path
import tempfile
from typing import Optional
import datetime as dt

try:
    import georinex as gr
    import xarray as xr
    GEORINEX_AVAILABLE = True
except Exception as e:
    print(f"Info: georinex/xarray not available ({e}); VTEC fetching will be disabled unless installed.")
    GEORINEX_AVAILABLE = False

try:
    # For .Z (LZW) compressed IONEX files commonly used by IGS/CODE
    from unlzw3 import unlzw
    UNLZW_AVAILABLE = True
except Exception as e:
    print(f"Info: unlzw3 not available ({e}); cannot decompress .Z IONEX files.")
    UNLZW_AVAILABLE = False
from datetime import datetime
from typing import List
from config import DATA_DIR

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

        # Public endpoints for indices (kept internal to avoid new CLI sources)
        self._swpc_f107_daily = "https://services.swpc.noaa.gov/json/f10cm_observed.json"
        # Monthly observed solar cycle indices (contains ssn and f10.7)
        self._swpc_cycle_indices = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
        # IONEX base (CDDIS requires Earthdata login). Try public mirrors first, then CDDIS.
        self._ionex_bases = [
            # JPL Sideshow public mirror (daily directories)
            "https://sideshow.jpl.nasa.gov/pub/iono_daily/{yyyy}/codg{doy:03d}0.{yy:02d}i.Z",
            "https://sideshow.jpl.nasa.gov/pub/iono_daily/{yyyy}/jplg{doy:03d}0.{yy:02d}i.Z",
            # AIUB/CODE public HTTPS (uppercase naming)
            "https://ftp.aiub.unibe.ch/CODE/{yyyy}/CODG{doy:03d}0.{yy:02d}I.Z",
            # CDDIS (may require authentication)
            "https://cddis.nasa.gov/archive/gnss/products/ionex/{yyyy}/{doy:03d}/codg{doy:03d}0.{yy:02d}i.Z",
            "https://cddis.nasa.gov/archive/gnss/products/ionex/{yyyy}/{doy:03d}/jplg{doy:03d}0.{yy:02d}i.Z",
        ]

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

    # -------------------- External Indices --------------------
    def fetch_f107(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch daily F10.7 (sfu) from NOAA/SWPC.

        Notes:
            - Tries daily endpoint first; if unavailable, falls back to
              monthly observed solar cycle indices and forward-fills to daily.
        Returns:
            DataFrame with columns: ['Time', 'F107'] (UTC dates at 00:00)
        """
        try:
            # Try daily observed JSON
            resp = requests.get(self._swpc_f107_daily, timeout=20)
            if resp.ok:
                data = resp.json()
                records = []
                for d in data or []:
                    # Common field names seen in SWPC responses
                    time_str = d.get('time_tag') or d.get('date') or d.get('timestamp')
                    flux = d.get('flux') or d.get('f10_7') or d.get('f107') or d.get('observed')
                    if time_str is None or flux is None:
                        continue
                    try:
                        t = pd.to_datetime(time_str).normalize()
                        f = float(flux)
                        records.append((t, f))
                    except Exception:
                        continue
                if records:
                    df = pd.DataFrame(records, columns=['Time', 'F107'])
                    df = df[(df['Time'] >= pd.to_datetime(start_dt).normalize()) & (df['Time'] <= pd.to_datetime(end_dt).normalize())]
                    df = df.sort_values('Time').reset_index(drop=True)
                    print(f"\n[OK] Retrieved {len(df)} F107 records from SWPC (daily)")
                    return df
        except Exception as e:
            print(f"[WARN] SWPC F107 daily endpoint failed: {e}")

        # Fallback: monthly observed solar cycle indices
        try:
            resp = requests.get(self._swpc_cycle_indices, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            records = []
            for d in data or []:
                time_str = d.get('time_tag') or d.get('time-tag') or d.get('date')
                f = d.get('f10.7') or d.get('f10_7') or d.get('observed_f10_7') or d.get('f107')
                if time_str is None or f is None:
                    continue
                try:
                    # Monthly values -> use first day of month
                    t = pd.to_datetime(time_str).to_period('M').to_timestamp()
                    records.append((t, float(f)))
                except Exception:
                    continue
            if not records:
                return pd.DataFrame()
            df_m = pd.DataFrame(records, columns=['Time', 'F107']).sort_values('Time')
            # Upsample to daily by forward-fill
            idx = pd.date_range(start=df_m['Time'].min(), end=df_m['Time'].max(), freq='D')
            df = df_m.set_index('Time').reindex(idx).ffill().reset_index().rename(columns={'index': 'Time'})
            df = df[(df['Time'] >= pd.to_datetime(start_dt).normalize()) & (df['Time'] <= pd.to_datetime(end_dt).normalize())]
            df = df.reset_index(drop=True)
            print(f"\n[OK] Retrieved {len(df)} F107 records from SWPC (monthly→daily)")
            return df
        except Exception as e:
            print(f"Failed to download F107 indices: {e}")
            return pd.DataFrame()

    def fetch_ssn(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch Sunspot Number (SSN) from SWPC monthly observed indices.

        Returns a daily series by forward-filling monthly values for convenience.
        Columns: ['Time', 'SSN']
        """
        try:
            resp = requests.get(self._swpc_cycle_indices, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            records = []
            for d in data or []:
                time_str = d.get('time_tag') or d.get('time-tag') or d.get('date')
                ssn = d.get('ssn') or d.get('sunspot_number') or d.get('ri')
                if time_str is None or ssn is None:
                    continue
                try:
                    t = pd.to_datetime(time_str).to_period('M').to_timestamp()
                    records.append((t, float(ssn)))
                except Exception:
                    continue
            if not records:
                return pd.DataFrame()
            df_m = pd.DataFrame(records, columns=['Time', 'SSN']).sort_values('Time')
            idx = pd.date_range(start=df_m['Time'].min(), end=df_m['Time'].max(), freq='D')
            df = df_m.set_index('Time').reindex(idx).ffill().reset_index().rename(columns={'index': 'Time'})
            df = df[(df['Time'] >= pd.to_datetime(start_dt).normalize()) & (df['Time'] <= pd.to_datetime(end_dt).normalize())]
            df = df.reset_index(drop=True)
            print(f"\n[OK] Retrieved {len(df)} SSN records from SWPC (monthly→daily)")
            return df
        except Exception as e:
            print(f"Failed to download SSN indices: {e}")
            return pd.DataFrame()

    # -------------------- VTEC from IONEX (CODE/JPL) --------------------
    def _download_ionex_for_day(self, day: datetime) -> Optional[Path]:
        """Download and decompress one day's IONEX (CODE/JPL) file.

        Returns local decompressed path (Path) or None if all sources fail.
        """
        yyyy = day.year
        yy = yyyy % 100
        doy = int(day.strftime('%j'))

        tmp_dir = Path(tempfile.gettempdir()) / "ionex_cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 1) Local fallback: check env var IONEX_LOCAL_DIR or project data/ionex
        local_base = os.environ.get('IONEX_LOCAL_DIR')
        candidate_dirs: List[Path] = []
        if local_base:
            candidate_dirs.append(Path(local_base))
        # Project data/ionex directory
        candidate_dirs.append(Path(DATA_DIR) / 'ionex')

        local_names = [
            f"codg{doy:03d}0.{yy:02d}i", f"jplg{doy:03d}0.{yy:02d}i",
            f"CODG{doy:03d}0.{yy:02d}I", f"JPLG{doy:03d}0.{yy:02d}I",
        ]
        for d in candidate_dirs:
            for name in local_names:
                for ext in ('', '.Z', '.z'):
                    p = d / name
                    if ext:
                        p = p.with_suffix(p.suffix + ext)
                    if p.exists():
                        # If compressed, decompress to temp dir
                        if p.suffix.lower().endswith('z'):
                            if not UNLZW_AVAILABLE:
                                print("[WARN] unlzw3 not installed; cannot read .Z IONEX. Please: pip install unlzw3")
                                break
                            try:
                                data = unlzw(p.read_bytes())
                                out_path = tmp_dir / p.stem  # drop .Z
                                with open(out_path, 'wb') as fout:
                                    fout.write(data)
                                print(f"[OK] Using local IONEX: {p}")
                                return out_path
                            except Exception as e:
                                print(f"[WARN] Failed to decompress local IONEX '{p}': {e}")
                                continue
                        else:
                            print(f"[OK] Using local IONEX: {p}")
                            return p

        for template in self._ionex_bases:
            url = template.format(yyyy=yyyy, yy=yy, doy=doy)
            try:
                resp = requests.get(url, timeout=60)
                if not resp.ok:
                    continue
                compressed_path = tmp_dir / Path(url).name
                with open(compressed_path, 'wb') as f:
                    f.write(resp.content)
                # Decompress .Z
                if compressed_path.suffix.lower() == '.Z' or compressed_path.suffix.lower() == '.z':
                    if not UNLZW_AVAILABLE:
                        print("[WARN] unlzw3 not installed; cannot read .Z IONEX. Please: pip install unlzw3")
                        return None
                    data = unlzw(compressed_path.read_bytes())
                    out_path = compressed_path.with_suffix('')  # drop .Z
                    with open(out_path, 'wb') as fout:
                        fout.write(data)
                    return out_path
                return compressed_path
            except Exception:
                continue
        return None

    def fetch_vtec(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch global average VTEC from IONEX (CODE/JPL).

        Output: DataFrame with columns ['Time','VTEC'].
        """
        if not GEORINEX_AVAILABLE:
            raise ImportError("georinex/xarray not installed. Please run: pip install georinex xarray unlzw3")

        all_frames: List[pd.DataFrame] = []
        day = start_dt.date()
        end_day = end_dt.date()

        while day <= end_day:
            local_path = self._download_ionex_for_day(datetime.combine(day, dt.time()))
            if local_path is None or not local_path.exists():
                print(f"[INFO] IONEX not available for {day}")
                day = (dt.datetime.combine(day, dt.time()) + dt.timedelta(days=1)).date()
                continue

            try:
                ds = gr.ionex(str(local_path))
                # Identify TEC variable name
                var_name = None
                if hasattr(ds, 'data_vars') and len(ds.data_vars) > 0:
                    var_name = list(ds.data_vars)[0]
                else:
                    var_name = 'tec'

                da = ds[var_name]
                # Filter to requested time range
                t0 = pd.to_datetime(start_dt)
                t1 = pd.to_datetime(end_dt)
                da_sel = da.sel(time=slice(t0, t1))
                if da_sel.size == 0:
                    day = (dt.datetime.combine(day, dt.time()) + dt.timedelta(days=1)).date()
                    continue

                # Calculate global average VTEC for each time step
                da_mean = da_sel.mean(dim=['lat', 'lon'])
                df_mean = da_mean.to_dataframe(name='VTEC').reset_index()
                df_mean = df_mean.rename(columns={'time': 'Time'})

                all_frames.append(df_mean[['Time', 'VTEC']])
                print(f"[OK] Global mean VTEC loaded for {day}: {len(df_mean)} points")
            except Exception as e:
                print(f"[WARN] Failed to parse IONEX for {day}: {e}")

            day = (dt.datetime.combine(day, dt.time()) + dt.timedelta(days=1)).date()

        if not all_frames:
            return pd.DataFrame()
        df = pd.concat(all_frames, axis=0).sort_values('Time').reset_index(drop=True)
        return df
