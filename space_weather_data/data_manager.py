"""Data management module"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from time_parser import TimeParser


class DataManager:
    """Manage local data files"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created data directory: {self.data_dir}")
    
    def save_to_csv(self, df: pd.DataFrame, filename: str, 
                    overwrite: bool = False) -> str:
        """Save DataFrame to CSV file"""
        if not filename.endswith('.csv'):
            filename = f"{filename}.csv"
        
        filepath = self.data_dir / filename
        
        if filepath.exists() and not overwrite:
            print(f"\n[INFO] File exists and overwrite is False: {filepath}")
            # The interactive input is removed to support non-interactive workflows.
            # The main script now handles this logic.
            return None # Indicate that save was skipped
        
        try:
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"[OK] Data saved: {filepath}")
            print(f"  - Records: {len(df)}")
            print(f"  - Columns: {len(df.columns)}")
            print(f"  - Size: {self._get_file_size(filepath)}")
            return str(filepath)
        except Exception as e:
            print(f"[ERROR] Save failed: {e}")
            return None
    
    def _get_file_size(self, filepath: Path) -> str:
        """Get human-readable file size"""
        size_bytes = filepath.stat().st_size
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def load_from_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from CSV file"""
        if not filename.endswith('.csv'):
            filename = f"{filename}.csv"
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            print(f"[OK] Data loaded: {filepath}")
            print(f"  - Records: {len(df)}")
            return df
        except Exception as e:
            print(f"[ERROR] Load failed: {e}")
            return None
    
    def list_data_files(self) -> List[str]:
        """List all data files"""
        csv_files = list(self.data_dir.glob("*.csv"))
        return [f.name for f in csv_files]
    
    def get_filename(self, start_dt: datetime, end_dt: datetime, 
                     param: str = None, **kwargs) -> str:
        """Generate standard filename"""
        
        time_label = TimeParser.get_time_range_label(start_dt, end_dt)
        source = kwargs.get('source', 'omniweb')

        # Build filename based on source
        if source == 'cdaweb':
            dataset = kwargs.get('dataset')
            if not dataset:
                raise ValueError("Dataset must be provided for cdaweb source")
            base_name = f"space_weather_{source}_{dataset}_{time_label}"
        elif source == 'goes':
            probe = kwargs.get('probe')
            instrument = kwargs.get('instrument')
            datatype = kwargs.get('datatype', '1min')
            if not all([probe, instrument]):
                raise ValueError("Probe and instrument must be provided for goes source")
            base_name = f"space_weather_{source}_{probe}_{instrument}_{datatype}_{time_label}"
        elif source == 'indices':
            base_name = f"space_weather_{source}_{time_label}"
        elif source == 'vtec':
            base_name = f"space_weather_{source}_{time_label}"
        else: # omniweb
            resolution = kwargs.get('resolution', 'hourly')
            base_name = f"space_weather_{source}_{resolution}_{time_label}"

        # Add parameter to the filename if provided
        if param:
            filename = f"{base_name}_{param}.csv"
        else:
            filename = f"{base_name}.csv"
            
        return filename
    
    def file_exists(self, filename: str) -> bool:
        """Check if data file exists"""
        if not filename.endswith('.csv'):
            filename = f"{filename}.csv"
        
        filepath = self.data_dir / filename
        return filepath.exists()
    
    def get_data_info(self) -> str:
        """Get data directory info summary"""
        files = self.list_data_files()
        
        if not files:
            return "Data directory is empty"
        
        info = f"\nData directory: {self.data_dir}\n"
        info += f"Total files: {len(files)}\n"
        info += "File list:\n"
        
        for i, filename in enumerate(files, 1):
            filepath = self.data_dir / filename
            size = self._get_file_size(filepath)
            modified = datetime.fromtimestamp(filepath.stat().st_mtime)
            info += f"  {i}. {filename} ({size}, {modified.strftime('%Y-%m-%d %H:%M')})\n"
        
        return info


if __name__ == "__main__":
    import numpy as np
    
    manager = DataManager()
    
    dates = pd.date_range('2023-06-15', periods=24, freq='h')
    df = pd.DataFrame({
        'Time': dates,
        'Bz': np.random.randn(24),
        'Vsw': 400 + np.random.randn(24) * 50,
        'nsw': 5 + np.random.randn(24) * 1,
    })
    
    filename = manager.get_filename(
        datetime(2023, 6, 15),
        datetime(2023, 6, 15),
        source="omniweb",
        resolution="hourly"
    )
    
    manager.save_to_csv(df, filename, overwrite=True)
    print(manager.get_data_info())
