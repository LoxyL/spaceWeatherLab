# Parameter Coverage (Sources, Resolution, Ranges)

This document summarizes the recommended sources, highest usable resolution, and
approximate earliest usable time ranges verified in the current environment.

Note: Hourly OMNI via pyspedas had parsing issues in our environment for early
years (e.g., 1963/1965). The ranges below refer to data availability in the
archives; if you encounter runtime issues, consider upgrading dependencies.

| Parameter | Source | Highest Resolution | Suggested Time Range | Notes |
|---|---|---|---|---|
| F10.7 | NOAA SWPC Observed (JSON) | Daily (monthly→daily fallback) | 1900-01 – present (daily via monthly forward-fill); stable daily ≥2004-10 | Works via `indices` source |
| SSN | NOAA SWPC Observed (JSON) | Daily (monthly→daily fallback) | 1749-01 – present (daily via monthly forward-fill) | Works via `indices` source |
| Dst | NASA OMNI (hourly) | Hourly | 1963 – present | pyspedas hourly parsing may need workaround |
| Kp | NASA OMNI (3-hour) | 3-hour | 1963 – present | Not in 1min files; prefer hourly/official Kp |
| B | NASA OMNI HRO | 1min | 1994-12 – present | Earlier months largely fill values |
| Bx | NASA OMNI HRO | 1min | 1994-12 – present | Verified 1994-12 |
| By | NASA OMNI HRO | 1min | 1994-12 – present | Verified 1994-12 |
| Bz | NASA OMNI HRO | 1min | 1994-12 – present | Verified 1994-12 |
| SPD (Vsw) | NASA OMNI HRO | 1min | 1994-12 – present | Verified 1994-12; coverage not always full |
| FP | Derived (from F10.7/SSN) | Daily | 1980 – 2024 (align to study span) | Available via `indices -p FP` |
| QI | Derived (from FP) | Daily | 1980 – 2024 (align to study span) | Available via `indices -p QI` |

Usage to fetch derived indices:

```bash
# FP and QI for Oct 2024
python main.py 2024-10 -s indices -p FP QI
```


