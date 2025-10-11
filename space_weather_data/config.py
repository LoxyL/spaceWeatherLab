"""Configuration: data sources and parameter mappings"""

OMNIWEB_BASE_URL = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

# Parameter mapping: user-friendly name -> OMNI column index
PARAMETER_MAPPING = {
    # OMNIWEB Aliases
    "B":      ["Bx", "By", "Bz"],
    "V":      "Vsw",
    "n":      "nsw",
    "T":      "Tsw",
    "p":      "Psw",
    "E":      "Esw",
    "beta":   "beta",
    "mach":   "Mach_num",
    "AE":     "AE",
    "AL":     "AL",
    "AU":     "AU",
    "symh":   "SYM-H",
    "symd":   "SYM-D",
    "asyh":   "ASYM-H",
    "asyd":   "ASYM-D",
    "pc":     "PC",
    "kp":     "Kp",
    "dst":    "DST",

    # GOES Vector aliases
    "b_gse": ["b_gse_0", "b_gse_1", "b_gse_2"],
}

DEFAULT_PARAMETERS = ["Bz", "Vsw", "nsw", "Psw", "Esw", "AE", "SYM-H", "ASYM-H", "PC"]

DATA_SOURCES = {
    "omniweb": {
        "name": "NASA OMNIweb",
        "url": OMNIWEB_BASE_URL,
        "description": "Solar wind and geomagnetic data from NASA",
        "resolution": ["1min", "5min", "hourly"],
    },
    "cdaweb": {
        "name": "NASA CDAWeb",
        "url": "https://cdaweb.gsfc.nasa.gov/",
        "description": "Access to a wide variety of space physics data",
        "resolution": [], # Resolution is dataset-specific
    },
    "goes": {
        "name": "NOAA GOES",
        "url": "https://www.goes.noaa.gov/",
        "description": "Geostationary Operational Environmental Satellites data",
        "resolution": ["1min", "5min"], # Example resolutions
    },
}

DEFAULT_GOES_PARAMETERS = {
    'mag': ['btotal', 'b_gse'],
    'xrs': ['xrsa', 'xrsb'],
    'particles': ['p1', 'p2', 'p3', 'e1', 'e2', 'e3']
}

DATA_DIR = "data"
CSV_FILENAME_FORMAT = "space_weather_{start}_{end}.csv"

DATE_FORMATS = {
    "year_range": r"^\d{4}-\d{4}$",
    "year": r"^\d{4}$",
    "month": r"^\d{4}-\d{2}$",
    "day": r"^\d{4}-\d{2}-\d{2}$",
}
