"""Data fetching module"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List


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
        else:
            print(f"Warning: {self.resolution} resolution requires CDF processing, using hourly instead")
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


if __name__ == "__main__":
    from time_parser import TimeParser
    
    parser = TimeParser()
    start, end = parser.parse("2023-06-15")
    
    fetcher = OMNIWebFetcher(resolution="hourly")
    df = fetcher.fetch(start, end, ["Bz", "Vsw", "nsw"])
    
    print(df.head())
