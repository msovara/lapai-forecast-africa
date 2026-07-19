@echo off
REM Monitor era5_n96_2020_2021.zarr build (no ExecutionPolicy change needed).
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0monitor_zarr_build.ps1" %*
