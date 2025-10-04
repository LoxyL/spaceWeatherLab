"""Configuration: data sources and parameter mappings"""

OMNIWEB_BASE_URL = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

# Parameter mapping: user-friendly name -> OMNI column index
PARAMETER_MAPPING = {
    # Magnetic field (GSM)
    "Bz": "23",
    "By": "22",
    "Bx": "21",
    "B": "20",
    
    # Solar wind
    "Vsw": "24",
    "nsw": "27",
    "Tsw": "28",
    
    # Pressure & energy
    "Psw": "29",
    "Esw": "35",
    
    # Geomagnetic indices
    "AE": "37",
    "AL": "38",
    "AU": "39",
    "SYM-H": "40",
    "SYM-D": "41",
    "ASYM-H": "42",
    "ASYM-D": "43",
    "PC": "44",
}

DEFAULT_PARAMETERS = ["Bz", "Vsw", "nsw", "Psw", "Esw", "AE", "SYM-H", "ASYM-H", "PC"]

DATA_SOURCES = {
    "omniweb": {
        "name": "NASA OMNIweb",
        "url": OMNIWEB_BASE_URL,
        "description": "Solar wind and geomagnetic data from NASA",
        "resolution": ["1min", "5min", "hourly"],
    },
}

DATA_DIR = "data"
CSV_FILENAME_FORMAT = "space_weather_{start}_{end}.csv"

DATE_FORMATS = {
    "year_range": r"^\d{4}-\d{4}$",
    "year": r"^\d{4}$",
    "month": r"^\d{4}-\d{2}$",
    "day": r"^\d{4}-\d{2}-\d{2}$",
}
