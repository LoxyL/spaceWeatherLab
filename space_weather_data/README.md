# Space Weather Data Acquisition Tool v0.4.0

A versatile command-line and Python tool to fetch, manage, and plot space weather data from various sources like NASA OMNIweb, CDAWeb, and NOAA GOES.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Data Sources](#data-sources)
  - [NASA OMNIweb (Default)](#nasa-omniweb-default)
  - [NASA CDAWeb](#nasa-cdaweb)
  - [NOAA GOES](#noaa-goes)
  - [Solar Indices (F10.7, SSN)](#solar-indices-f107-ssn)
  - [IGS GIM VTEC (IONEX)](#igs-gim-vtec-ionex)
- [Command Line Options](#command-line-options)
- [Usage Examples](#usage-examples)
- [Python API](#python-api)
- [Data Format](#data-format)
- [Notes](#notes)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Links](#links)

## Features

-   Fetch data from multiple sources: NASA OMNIweb, CDAWeb, and NOAA GOES.
-   Flexible time range input (day, month, year, or date range).
-   Command-line interface for easy scripting and data retrieval.
-   Saves data locally in clean, per-parameter CSV files.
-   Automatic plotting of the fetched data.
-   Python API for integration into other projects.

## Quick Start

```bash
# Get 5-minute OMNI data for a specific day and plot it
python main.py 2023-06-15 -r 5min --plot

# Get GOES-15 X-ray flux data
python main.py 2013-03-17 -s goes --probe 15 --instrument xrs

# Get ACE magnetic field data from CDAWeb
python main.py 2004-11-07 -s cdaweb --dataset AC_H0_MFI -p BZ_GSE
```

## Data Sources

### NASA OMNIweb (Default)

A compiled dataset providing a continuous, gap-filled solar wind record at 1 AU. It is created by combining and cross-calibrating data from multiple upstream satellites (like ACE, Wind, IMP 8). Ideal for general long-term analysis.

#### OMNI High-Resolution (1-min, 5-min)

| | |
|---|---|
| **Time Range** | **1981 - Present** |
| **Resolutions**| `1min`, `5min` |
| **Sources** | `high_res_omni` (1981-2017, ASCII), `hro`/`hro2` (1995-Present, CDF). The tool automatically handles fallback. |
| **Parameters** | `B`, `Bx`, `By`, `Bz`, `By_GSE`, `Bz_GSE`, `Vsw`, `nsw`, `Tsw`, `Psw`, `Esw`, `Vx`, `Vy`, `Vz`, `beta`, `Mach_num`, `AE`, `AL`, `AU`, `SYM-H`, `SYM-D`, `ASY-H`, `ASY-D`, `PC`, `Kp`, `DST` |

#### OMNI Hourly Resolution

| | |
|---|---|
| **Time Range** | **1963 - Present** |
| **Resolution** | `hourly` |
| **Source** | `low_res_omni` ASCII files |
| **Parameters** | `B`, `Bx`, `By`, `Bz`, `Vsw`, `nsw`, `Tsw`, `Psw`, `Esw`, `AE`, `AL`, `AU`, `DST`, `PC` |

### NASA CDAWeb

A vast archive of original, high-cadence data from numerous spacecraft missions. Ideal for specific event studies or accessing raw data not available in the compiled OMNI dataset.

**You must find the Dataset ID and Variable Names on the CDAWeb site yourself.**

#### How to find Datasets & Variables?

1.  Go to the [CDAWeb Data Browser](https://cdaweb.gsfc.nasa.gov/index.html/).
2.  Select a mission (e.g., `Wind`) and instrument (e.g., `Magnetic Fields (space)`).
3.  Find the **Dataset ID** (e.g., `WI_H0_MFI`) and the **Variable Names** inside it (e.g., `BGSE`).

#### Example CDAWeb Datasets

| Dataset ID         | Mission/Instrument | Example Variables          | Approx. Resolution | Approx. Time Range |
| :----------------- | :----------------- | :------------------------- | :----------------- | :----------------- |
| `WI_H0_MFI`        | Wind / MFI         | `BGSE` (vector)            | ~3 seconds         | 1994 - Present     |
| `AC_H0_MFI`        | ACE / MFI          | `BGSE` (vector)            | 16 seconds         | 1997 - Present     |
| `AC_H1_SWE`        | ACE / SWEPAM       | `Vp`, `Np`                 | 64 seconds         | 1997 - Present     |
| `MMS1_FGM_SRVY_L2` | MMS / FGM          | `mms1_fgm_b_gse_srvy_l2`   | ~4.5 seconds       | 2015 - Present     |

### NOAA GOES

Provides access to Geostationary Operational Environmental Satellites data, crucial for space weather monitoring.

| Instrument  | Description               | Example Variables (Probe 15) | Approx. Time Range (GOES 8-15) |
| :---------- | :------------------------ | :--------------------------- | :----------------------------- |
| `mag`       | Magnetic Field (1-min)    | `btotal`, `b_gse`    | ~1995 - ~2020                  |
| `particles` | Energetic Particles (1/5-min) | `p1`, `p2`, `p3`, `e1`, `e2`, `e3`   | ~1995 - ~2020                  |
| `xrs`       | Solar X-Ray Flux (1-min)  | `xrsa`, `xrsb`               | ~1995 - ~2020                  |

- **Supported Probes**: GOES satellites `8` through `15`. The exact time range depends on the specific probe's operational window.

### Solar Indices (F10.7, SSN)

Daily or monthly observed indices fetched from NOAA SWPC public JSON services.

| Index  | Description                 | Columns | Approx. Time Range | Resolution | Source |
| :----- | :-------------------------- | :------ | :------------------ | :-------- | :----- |
| F10.7  | 10.7 cm solar radio flux    | `F107`  | 2004-10 – Present   | Daily (from monthly observed, forward-filled) | SWPC |
| SSN    | International Sunspot Number| `SSN`   | 1749-01 – Present   | Daily (from monthly observed, forward-filled) | SWPC |

### IGS GIM VTEC (IONEX)

Global ionospheric maps parsed from IONEX files (CODE/JPL). The loader now returns the global mean VTEC per epoch with columns `Time, VTEC` (unweighted mean across all grid cells).

| | |
|---|---|
| **Approx. Time Range** | 1998 – Present (CODE/CODG; JPL/JPLG similar; occasional gaps) |
| **Temporal Resolution** | 2 hours nominal (00:00, 02:00, ..., 22:00 UTC); some centers offer 1-hour or 15-min in recent years |
| **Spatial Grid** | ~2.5° latitude × 5° longitude |
| **Columns** | `Time`, `VTEC` |
| **Formats** | Input: IONEX (`*.i`, often compressed `*.Z`) |

## Command Line Options

| Option              | Short | Description                                                               |
|---------------------|-------|---------------------------------------------------------------------------|
| `time_input`        |       | Time range (YYYY-MM-DD, YYYY-MM, YYYY-MM-DD to YYYY-MM-DD)                 |
| `--parameters`      | `-p`  | Parameters/variables to fetch (e.g., `BZ_GSE`, `Vsw`). Optional for GOES. |
| `--source`          | `-s`  | Data source (`omniweb`, `cdaweb`, `goes`). Default: `omniweb`.             |
|                       |       | Also supports `indices` (F10.7/SSN) and `vtec` (IGS GIM).                 |
| `--resolution`      | `-r`  | Time resolution (`hourly`, `5min`, `1min`). For OMNI.                      |
| `--dataset`         |       | CDAWeb dataset ID (e.g., `WI_H0_MFI`). **Required for `cdaweb`**.           |
| `--probe`           |       | GOES satellite probe number (e.g., `15`). **Required for `goes`**.          |
| `--instrument`      |       | GOES instrument (`mag`, `particles`, `xrs`). **Required for `goes`**.     |
| `--datatype`        |       | GOES data type (e.g., `1min`, `ep8`). Default: `1min`.                     |
| `--output`          | `-o`  | Custom output filename.                                                   |
| `--overwrite`       |       | Overwrite existing data files.                                            |
| `--plot`            |       | Generate and display a plot of the data.                                  |
| `--plot-file`       |       | Save the plot to a specified file.                                        |

## Usage Examples

### Fetching OMNI Data
```bash
# Download hourly OMNI data for a specific month
python main.py 2023-06 -p Bz Vsw nsw

# Download 5-minute resolution OMNI data and plot it
python main.py 2023-06-15 -r 5min --plot
```

### Fetching CDAWeb Data
```bash
# Get ACE magnetic field data for a specific day
python main.py 2004-11-07 -s cdaweb --dataset AC_H0_MFI -p BZ_GSE
```

### Fetching GOES Data
```bash
# Get GOES-15 one-minute X-ray flux data for a specific day
python main.py 2013-03-17 -s goes --probe 15 --instrument xrs

# Get GOES-13 8-second magnetometer data and plot it
python main.py 2012-07-12 -s goes --probe 13 --instrument mag --datatype ep8 --plot
```

### Fetching Solar Indices (Python API)
```python
from space_weather_data.time_parser import TimeParser
from space_weather_data.data_fetcher import DataFetcher

parser = TimeParser()
start_idx, end_idx = parser.parse("2024-10")
fetcher = DataFetcher()

# F10.7 daily series
f107_df = fetcher.fetch_f107(start_idx, end_idx)

# Sunspot Number daily (monthly observed forward-filled)
ssn_df = fetcher.fetch_ssn(start_idx, end_idx)
```

### Fetching Solar Indices (CLI)
```bash
# Fetch F10.7 and SSN for 2024-10 and save as CSV (one column each)
python main.py 2024-10 -s indices -p F107 SSN
```

### Fetching VTEC (IONEX, CLI)
```bash
# Fetch global-mean VTEC for 2024-10-23 (CSV: Time,VTEC)
python main.py 2024-10-23 -s vtec -p VTEC
```

### Fetching VTEC (IONEX, Python API)
```python
from space_weather_data.time_parser import TimeParser
from space_weather_data.data_fetcher import DataFetcher

parser = TimeParser()
start, end = parser.parse("2024-10-23")
fetcher = DataFetcher()
vtec_df = fetcher.fetch_vtec(start, end)  # Time, VTEC (global mean)
```

### VTEC Output Format
Returns two columns `Time, VTEC`, where `VTEC` is the global mean at each epoch (simple average across all grid cells).

## Python API

```python
from space_weather_data.time_parser import TimeParser
from space_weather_data.data_fetcher import DataFetcher
from space_weather_data.data_manager import DataManager

# --- OMNI Example ---
parser = TimeParser()
start, end = parser.parse("2023-06")
fetcher = DataFetcher(resolution="hourly")
omni_df = fetcher.fetch_omni(start, end)
# (omni_df will contain all parameters, you can filter it with pandas)

# --- CDAWeb Example ---
start_cda, end_cda = parser.parse("2010-01-01")
cda_df = fetcher.fetch_cdaweb(
    dataset="WI_H0_MFI", 
    start_dt=start_cda, 
    end_dt=end_cda, 
    parameters=["BZ_GSE"] # Uses the parameter mapping
)

# --- GOES Example ---
start_goes, end_goes = parser.parse("2013-03-17")
goes_df = fetcher.fetch_goes(
    probe='15',
    start_dt=start_goes,
    end_dt=end_goes,
    instrument='xrs',
    datatype='1min'
)

# --- Solar Indices ---
start_idx, end_idx = parser.parse("2024-10")
f107_df = fetcher.fetch_f107(start_idx, end_idx)
ssn_df = fetcher.fetch_ssn(start_idx, end_idx)

# --- Saving Data ---
manager = DataManager()
# (Example for saving one parameter from the OMNI dataframe)
filename = manager.get_filename(
    start, end, param="Bz", source="omniweb", resolution="hourly"
)
manager.save_to_csv(omni_df[['Time', 'Bz']], filename)
```

## Data Format

Data is saved as **one CSV file per parameter**.

**Filename Format**:
- OMNI: `space_weather_omniweb_{resolution}_{time_label}_{parameter}.csv`
- CDAWeb: `space_weather_cdaweb_{dataset}_{time_label}_{parameter}.csv`
- GOES: `space_weather_goes_{probe}_{instrument}_{datatype}_{time_label}_{parameter}.csv`
- Indices (F10.7/SSN): `space_weather_indices_{time_label}_{parameter}.csv`
- VTEC: `space_weather_vtec_{time_label}_{parameter}.csv`

**Example Content (`..._Bz.csv`)**:
```csv
Time,Bz
2023-06-15 00:00:00,0.0
...
```

## Notes

- All data is in UTC. Missing values are stored as NaN.
- The `pyspedas` library caches downloaded data locally, usually in a folder named `pyspedas_data` in your home directory.

## Project Structure

```
space_weather_data/
├── main.py              # CLI entry point
├── data_fetcher.py      # Unified data downloader
├── time_parser.py       # Time input parser
├── data_manager.py      # CSV file manager
├── plotter.py           # Data plotter
├── config.py            # Configuration
├── requirements.txt     # Dependencies
└── data/                # Saved CSV data
```

## Requirements

- Python 3.7+
- pandas, numpy, matplotlib
- pyspedas (OMNI/CDAWeb/GOES)
- cdflib
- georinex, xarray, unlzw3 (only VTEC/IONEX)

## Links

- [NASA OMNIweb](https://omniweb.gsfc.nasa.gov/)
- [NASA CDAWeb](https://cdaweb.gsfc.nasa.gov/)
- [NOAA GOES](https://www.goes.noaa.gov/)

---

**Version**: 0.4.0  
**License**: MIT