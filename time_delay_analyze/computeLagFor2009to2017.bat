@echo off
setlocal enabledelayedexpansion

REM Change to the directory of this script to ensure relative paths work
pushd "%~dp0"

for /L %%Y in (2009,1,2017) do (
	echo ================================================================
	echo Computing lags for %%Y
	python export_daily_delays.py --time %%Y-01-01/%%Y-12-31 --out-csv lags_%%Y.csv
	echo Done computing lags for %%Y
	echo ================================================================
)

popd

endlocal


