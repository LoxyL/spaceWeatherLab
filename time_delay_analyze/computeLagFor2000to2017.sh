#!/bin/bash
for year in {2000..2017}; do
    echo "================================================"
    echo "Computing lags for $year"
    python export_daily_delays.py --time $year-01-01/$year-12-31 --out-csv lags_$year.csv
    echo "Done computing lags for $year"
    echo "================================================"
done