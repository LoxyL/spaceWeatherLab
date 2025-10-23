# All Available Parameters (from README)

This document summarizes all fetchable parameters described in `space_weather_data/README.md`. The data source is now part of the table instead of headings.

| Parameter(s) / Dataset | Source | Resolution | Approx. Time Range | Notes |
|---|---|---|---|---|
| **B**, **Bx**, **By**, **Bz**, By_GSE, Bz_GSE | NASA OMNIweb (HRO/HRO2, CDF) | 1min / 5min | 1981 – Present | Magnetic field (high-res) |
| **Vsw**, **nsw**, Tsw, **Psw**, **Esw**, Vx, Vy, Vz | NASA OMNIweb (HRO/HRO2, CDF) | 1min / 5min | 1981 – Present | Solar wind plasma (high-res) |
| beta, Mach_num | NASA OMNIweb (HRO/HRO2, CDF) | 1min / 5min | 1981 – Present | Derived plasma params |
| **AE**, **AL**, **AU**, SYM-H, SYM-D, ASY-H, ASY-D, PC | NASA OMNIweb (HRO/HRO2, CDF) | 1min / 5min | 1981 – Present | Geomagnetic indices (high-res) |
| **Kp**, **DST** | NASA OMNIweb (HRO/HRO2 listing) |  | 1981 – Present | Canonical cadence is 3-hour (Kp) and hourly (Dst) |
| B, Bx, By, Bz | NASA OMNIweb (low-res ASCII) | hourly | 1963 – Present | Magnetic field (hourly) |
| Vsw, nsw, Tsw, Psw, Esw | NASA OMNIweb (low-res ASCII) | hourly | 1963 – Present | Solar wind plasma (hourly) |
| AE, AL, AU, DST, PC | NASA OMNIweb (low-res ASCII) | hourly | 1963 – Present | Geomagnetic indices (hourly) |
| btotal, b_gse (mag) | NOAA GOES (NCEI) | 1min (example ep8 also used) | ~1995 – ~2020 | Magnetometer |
| p1, p2, p3, e1, e2, e3 (particles) | NOAA GOES (NCEI) | 1min / 5min | ~1995 – ~2020 | Energetic particles |
| **xrsa**, **xrsb** (**xrs**) | NOAA GOES (NCEI) | 1min | ~1995 – ~2020 | X-ray flux |
| **F10.7** | NOAA SWPC Observed (JSON) | Daily (from monthly observed, forward-filled) | 2004-10 – Present (daily endpoint); monthly observed available | 10.7 cm solar radio flux |
| **SSN** | NOAA SWPC Observed (JSON) | Daily (from monthly observed, forward-filled) | 1749-01 – Present | International Sunspot Number |
| **VTEC** (global mean per epoch) | IGS GIM (IONEX, CODE/JPL) | typically 2 hours | ~1998 – Present (varies by center) | Project returns global mean per epoch |
| WI_H0_MFI (Wind / MFI → BGSE) | NASA CDAWeb | ~3 s | 1994 – Present | Magnetic field |
| **AC_H0_MFI** (ACE / MFI → BGSE) | NASA CDAWeb | 16 s | 1997 – Present | Magnetic field |
| **AC_H1_SWE** (ACE / SWEPAM → Vp, Np) | NASA CDAWeb | 64 s | 1997 – Present | Plasma |
| MMS1_FGM_SRVY_L2 (MMS / FGM → mms1_fgm_b_gse_srvy_l2) | NASA CDAWeb | ~4.5 s | 2015 – Present | Magnetic field |

Notes:
- Time ranges and cadences are taken directly from the README; some archives may have gaps.
- For operational fetching, use the CLI in `space_weather_data/main.py` with the appropriate `--source`, `--resolution`, `--dataset`, `--probe`, `--instrument`, and `--datatype` flags, as shown in README examples.
