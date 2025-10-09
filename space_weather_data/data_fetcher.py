"""Data fetching module"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
import tempfile
import os

try:
    import cdflib
    CDF_AVAILABLE = True
except ImportError:
    CDF_AVAILABLE = False


class OMNIWebFetcher:
    """OMNI web data fetcher"""
    
    def __init__(self, resolution: str = "hourly"):
        self.resolution = resolution
        self.data_urls = {
            "hourly": "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat",
            "5min": "https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro_5min/{year}/omni_hro_5min_{year}{month:02d}01_v01.cdf",
            "1min": "https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro_1min/{year}/omni_hro_1min_{year}{month:02d}01_v01.cdf",
        }
    
    def fetch(self, start_dt: datetime, end_dt: datetime, 
              parameters: List[str]) -> pd.DataFrame:
        """Fetch data from NASA OMNI database"""
        print(f"\nFetching data from NASA OMNI database...")
        print(f"Time range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        print(f"Resolution: {self.resolution}")
        
        if self.resolution == "hourly":
            return self._fetch_hourly_data(start_dt, end_dt, parameters)
        elif self.resolution in ["5min", "1min"]:
            if not CDF_AVAILABLE:
                print(f"Warning: cdflib not installed. Please run: pip install cdflib")
                print(f"Falling back to hourly resolution")
                return self._fetch_hourly_data(start_dt, end_dt, parameters)
            return self._fetch_high_res_data(start_dt, end_dt, parameters)
        else:
            print(f"Warning: Unknown resolution '{self.resolution}', using hourly instead")
            return self._fetch_hourly_data(start_dt, end_dt, parameters)
    
    def _fetch_hourly_data(self, start_dt: datetime, end_dt: datetime,
                           parameters: List[str]) -> pd.DataFrame:
        """Fetch hourly resolution data"""
        all_data = []
        
        for year in range(start_dt.year, end_dt.year + 1):
            try:
                year_data = self._download_year_data(year)
                all_data.append(year_data)
                print(f"[OK] Year {year} downloaded ({len(year_data)} records)")
            except Exception as e:
                print(f"[ERROR] Year {year} failed: {e}")
        
        if not all_data:
            raise Exception("No data downloaded")
        
        df = pd.concat(all_data, ignore_index=True)
        df = df[(df['Time'] >= start_dt) & (df['Time'] <= end_dt)]
        
        columns_to_keep = ['Time'] + [p for p in parameters if p in df.columns]
        df = df[columns_to_keep]
        
        print(f"\n[OK] Retrieved {len(df)} records")
        return df
    
    def _download_year_data(self, year: int) -> pd.DataFrame:
        """Download data for a specific year"""
        url = self.data_urls["hourly"].format(year=year)
        print(f"Downloading: {url}")
        
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        return self._parse_omni2_hourly(response.text, year)
    
    def _parse_omni2_hourly(self, text: str, year: int) -> pd.DataFrame:
        """
        Parse OMNI2 hourly data format (space-separated ASCII)
        
        Column indices (0-based):
        0: Year, 1: Day, 2: Hour
        8: |B|, 12: Bx GSM, 15: By GSM, 16: Bz GSM
        22: Proton Temp, 23: Proton Density, 24: Plasma Speed
        28: Flow Pressure, 35: Electric Field
        40: DST, 41: AE, 51: PC(N), 52: AL, 53: AU
        """
        lines = text.strip().split('\n')
        data_records = []
        
        for line in lines:
            if not line.strip():
                continue
            
            try:
                parts = line.split()
                if len(parts) < 30:
                    continue
                
                year_val = int(parts[0])
                day_of_year = int(parts[1])
                hour = int(parts[2])
                dt = datetime(year_val, 1, 1) + timedelta(days=day_of_year-1, hours=hour)
                
                record = {
                    'Time': dt,
                    'B': self._parse_float(parts, 8),
                    'Bx': self._parse_float(parts, 12),
                    'By': self._parse_float(parts, 15),
                    'Bz': self._parse_float(parts, 16),
                    'Tsw': self._parse_float(parts, 22),
                    'nsw': self._parse_float(parts, 23),
                    'Vsw': self._parse_float(parts, 24),
                    'Psw': self._parse_float(parts, 28),
                    'Esw': self._parse_float(parts, 35),
                    'DST': self._parse_float(parts, 40),
                    'AE': self._parse_float(parts, 41),
                    'PC': self._parse_float(parts, 51),
                    'AL': self._parse_float(parts, 52),
                    'AU': self._parse_float(parts, 53),
                }
                
                record['SYM-H'] = record['DST']
                record['ASYM-H'] = pd.NA
                
                data_records.append(record)
                
            except (ValueError, IndexError):
                continue
        
        return pd.DataFrame(data_records)
    
    def _parse_float(self, parts: list, index: int) -> float:
        """Parse float from split line, handling missing values"""
        try:
            if index >= len(parts):
                return pd.NA
            
            value_str = parts[index].strip()
            if not value_str:
                return pd.NA
            
            value = float(value_str)
            
            if abs(value) >= 9999.0:
                return pd.NA
            if abs(value - 999.9) < 0.1 or abs(value - 99.9) < 0.1:
                return pd.NA
            if abs(value - 9.999) < 0.001 or abs(value - 99.99) < 0.01:
                return pd.NA
                
            return value
        except (ValueError, IndexError):
            return pd.NA
    
    def _fetch_high_res_data(self, start_dt: datetime, end_dt: datetime,
                             parameters: List[str]) -> pd.DataFrame:
        """Fetch high resolution (5min or 1min) data from CDF files"""
        all_data = []
        
        # Generate list of months to download
        current_dt = datetime(start_dt.year, start_dt.month, 1)
        end_month = datetime(end_dt.year, end_dt.month, 1)
        
        while current_dt <= end_month:
            try:
                month_data = self._download_cdf_month_data(current_dt.year, current_dt.month)
                if month_data is not None and not month_data.empty:
                    all_data.append(month_data)
                    print(f"[OK] {current_dt.strftime('%Y-%m')} downloaded ({len(month_data)} records)")
            except Exception as e:
                print(f"[WARNING] {current_dt.strftime('%Y-%m')} failed: {e}")
            
            # Move to next month
            if current_dt.month == 12:
                current_dt = datetime(current_dt.year + 1, 1, 1)
            else:
                current_dt = datetime(current_dt.year, current_dt.month + 1, 1)
        
        if not all_data:
            raise Exception("No data downloaded")
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Filter by date range
        df = df[(df['Time'] >= start_dt) & (df['Time'] <= end_dt + timedelta(days=1))]
        
        # Select requested columns
        columns_to_keep = ['Time'] + [p for p in parameters if p in df.columns]
        df = df[columns_to_keep]
        
        print(f"\n[OK] Retrieved {len(df)} records")
        return df
    
    def _download_cdf_month_data(self, year: int, month: int) -> pd.DataFrame:
        """Download CDF data for a specific month"""
        url = self.data_urls[self.resolution].format(year=year, month=month)
        print(f"Downloading: {url}")
        
        # Download CDF file to temporary location
        response = requests.get(url, timeout=120)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.cdf') as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        try:
            # Parse CDF file
            df = self._parse_cdf_file(tmp_path)
            return df
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def _parse_cdf_file(self, cdf_path: str) -> pd.DataFrame:
        """Parse OMNI CDF file
        
        CDF variable names mapping:
        - Epoch: Time
        - BX_GSE, BY_GSM, BZ_GSM: Magnetic field components
        - Vx, Vy, Vz or flow_speed: Solar wind velocity
        - proton_density: Proton density
        - T: Temperature
        - Pressure: Flow pressure
        - E: Electric field
        - AE_INDEX, SYM_H, etc.: Geomagnetic indices
        """
        cdf_file = cdflib.CDF(cdf_path)
        
        # Get all available variables
        cdf_info = cdf_file.cdf_info()
        variables = cdf_info.zVariables
        
        data_records = {}
        
        # Time variable (Epoch)
        if 'Epoch' in variables:
            epochs = cdf_file.varget('Epoch')
            # Convert CDF epoch to datetime
            times = cdflib.cdfepoch.to_datetime(epochs)
            data_records['Time'] = times
        else:
            raise Exception("No Epoch variable found in CDF file")
        
        # CDF variable name mapping to standard names
        cdf_mapping = {
            # Magnetic field
            'BX_GSE': 'Bx',
            'BY_GSM': 'By', 
            'BZ_GSM': 'Bz',
            'B': 'B',
            'ABS_B': 'B',
            
            # Solar wind
            'flow_speed': 'Vsw',
            'Vx': 'Vx',
            'Vy': 'Vy', 
            'Vz': 'Vz',
            'proton_density': 'nsw',
            'T': 'Tsw',
            'Pressure': 'Psw',
            'E': 'Esw',
            
            # Indices
            'AE_INDEX': 'AE',
            'AL_INDEX': 'AL',
            'AU_INDEX': 'AU',
            'SYM_H': 'SYM-H',
            'SYM_D': 'SYM-D',
            'ASY_H': 'ASYM-H',
            'ASY_D': 'ASYM-D',
            'PC_N_INDEX': 'PC',
        }
        
        # Extract variables
        for cdf_var, std_var in cdf_mapping.items():
            if cdf_var in variables:
                try:
                    values = cdf_file.varget(cdf_var)
                    # Handle fill values
                    values = self._clean_cdf_values(values)
                    data_records[std_var] = values
                except Exception as e:
                    print(f"Warning: Failed to read {cdf_var}: {e}")
        
        df = pd.DataFrame(data_records)
        return df
    
    def _clean_cdf_values(self, values):
        """Clean CDF values, replacing fill values with NA"""
        # Convert to numpy array
        values = np.array(values)
        
        # Common fill values in OMNI data
        fill_values = [
            -1.0e31, 9.9692099683868690e+36, 
            999.99, 99999.0, 9999.99, 999.9, 99.99, 9.999
        ]
        
        # Replace fill values with NA
        for fill_val in fill_values:
            if np.issubdtype(values.dtype, np.floating):
                values = np.where(np.abs(values - fill_val) < abs(fill_val * 0.01) if fill_val != 0 else np.abs(values - fill_val) < 0.01, 
                                np.nan, values)
        
        # Replace extremely large values
        if np.issubdtype(values.dtype, np.floating):
            values = np.where(np.abs(values) > 1e10, np.nan, values)
        
        return values


if __name__ == "__main__":
    from time_parser import TimeParser
    
    parser = TimeParser()
    start, end = parser.parse("2023-06-15")
    
    fetcher = OMNIWebFetcher(resolution="hourly")
    df = fetcher.fetch(start, end, ["Bz", "Vsw", "nsw"])
    
    print(df.head())
