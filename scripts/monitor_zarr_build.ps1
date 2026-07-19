# Monitor era5_n96_2020_2021.zarr build progress (Windows PowerShell only).
param(
    [string]$RepoRoot = (Split-Path $PSScriptRoot -Parent),
    [int]$IntervalSec = 60
)
$zarr = Join-Path $RepoRoot "data/processed/lapai/era5_n96_2020_2021.zarr"
$stderr = Join-Path $RepoRoot "reports/zarr_build_stderr.log"
$debug = Join-Path $RepoRoot "reports/zarr_build_debug.log"
$log = if (Test-Path $debug) { $debug } else { $stderr }
Write-Host "Watching $zarr (Ctrl+C to stop)"
while ($true) {
    $chunks = 0
    $mb = 0
    if (Test-Path $zarr) {
        $chunks = (Get-ChildItem "$zarr/data" -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notlike '.*' }).Count
        $mb = [math]::Round(
            (Get-ChildItem $zarr -Recurse -File -ErrorAction SilentlyContinue |
                Measure-Object Length -Sum).Sum / 1MB, 1)
    }
    $tail = if (Test-Path $log) {
        (Get-Content $log -Tail 1 -ErrorAction SilentlyContinue)
    } else { "(no build log)" }
    $alive = [bool](Get-Process anemoi-datasets -ErrorAction SilentlyContinue)
    Write-Host ("[{0}] alive={1} data_chunks={2} size_mb={3}" -f (Get-Date -Format 'HH:mm:ss'), $alive, $chunks, $mb)
    if ($tail) { Write-Host "  $tail" }
    Start-Sleep -Seconds $IntervalSec
}
