# Space Weather Data Acquisition Tool

Download solar wind and geomagnetic data from NASA OMNIweb.

## Features

- **Real data from NASA** - Direct download from OMNI database (1963-present)
- **Flexible time input** - Year, month, day, or date range
- **Customizable parameters** - Select which data to download
- **CSV export** - Easy to analyze with Excel/Python
- **Built-in plotting** - Visualize data instantly

## Quick Start

### Installation

```bash
cd space_weather_data
pip install -r requirements.txt
```

### Usage

```bash
# Download 2023 data (default parameters)
python main.py 2023

# Specify parameters
python main.py 2023-06 -p Bz Vsw nsw AE SYM-H

# Download data for a specific time resolution
python main.py 2023-06-15 -r 5min

# Plot data
python main.py 2023-06-15 --plot

# List local files
python main.py --list

# Show available parameters
python main.py --show-params
```

## Supported Parameters

### Magnetic Field (GSM)
- **Bz, By, Bx** - IMF components (nT)
- **B** - Field magnitude (nT)

### Solar Wind
- **Vsw** - Speed (km/s)
- **nsw** - Proton density (N/cm³)
- **Tsw** - Temperature (K)

### Pressure & Energy
- **Psw** - Dynamic pressure (nPa)
- **Esw** - Electric field (mV/m)

### Geomagnetic Indices
- **AE, AL, AU** - Auroral electrojet (nT)
- **SYM-H** - Symmetric ring current (like Dst)
- **ASYM-H** - Asymmetric ring current
- **PC** - Polar cap index

## Time Format

- `2023` - Full year
- `2023-06` - Full month
- `2023-06-15` - Single day
- `2020-2023` - Year range

## Command Line Options

```
python main.py <time> [options]

Positional:
  time                  Time input (e.g., 2023, 2023-06, 2023-06-15)

Options:
  -p, --parameters      Parameters to download (default: Bz Vsw nsw Psw Esw AE SYM-H ASYM-H PC)
  -r, --resolution      Time resolution: hourly (default), 5min, 1min
  -o, --output          Output filename
  --overwrite           Overwrite existing file
  --plot                Show plot
  --list                List local data files
  --show-params         Show available parameters
```

## Python API

```python
from time_parser import TimeParser
from data_fetcher import OMNIWebFetcher
from data_manager import DataManager

# Parse time
parser = TimeParser()
start_dt, end_dt = parser.parse("2023-06")

# Fetch data
fetcher = OMNIWebFetcher(resolution="hourly")
df = fetcher.fetch(start_dt, end_dt, ["Bz", "Vsw", "AE"])

# Save data
manager = DataManager()
manager.save_to_csv(df, "my_data.csv")
```

## Data Format

CSV files with UTC timestamps:

```
Time,Bz,Vsw,nsw,AE
2023-06-15 00:00:00,0.0,323.0,3.5,58.0
2023-06-15 01:00:00,0.5,330.0,3.4,62.0
...
```

Filename: `space_weather_omniweb_hourly_YYYYMMDD.csv`

## Data Source

- **Source**: NASA OMNI database
- **URL**: https://spdf.gsfc.nasa.gov/pub/data/omni/
- **Resolution**: 1 hour (recommended)
- **Coverage**: 1963 - present
- **Time zone**: UTC
- **Missing values**: NaN

## Notes

- Data downloaded directly from NASA, no API key needed
- Some time periods may have missing data
- SYM-H uses DST index as substitute (similar physical meaning)
- ASYM-H not available in basic OMNI2 dataset

## Project Structure

```
space_weather_data/
├── main.py              # CLI entry point
├── data_fetcher.py      # NASA data downloader
├── time_parser.py       # Time input parser
├── data_manager.py      # CSV file manager
├── plotter.py           # Data plotter
├── config.py            # Parameter configuration
├── requirements.txt     # Dependencies
└── data/                # Downloaded data
```

## Requirements

- Python 3.7+
- pandas
- numpy
- requests
- matplotlib

## Links

- [NASA OMNIweb](https://omniweb.gsfc.nasa.gov/)
- [CDAWeb](https://cdaweb.gsfc.nasa.gov/)

---

**Version**: 1.0.0  
**License**: MIT
