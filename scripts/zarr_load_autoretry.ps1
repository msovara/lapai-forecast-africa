$ErrorActionPreference = "Continue"
$exe = "C:\Users\MthethoSovara\anaconda3\envs\lapai-anemoi\Scripts\anemoi-datasets.exe"
$repo = "C:\Users\MthethoSovara\tiny-media-analysis\lapai-forecast"
$zarr = "data\processed\lapai\era5_n96_2020_2021.zarr"
$wrapperLog = Join-Path $repo "reports\zarr_build_retry.log"
$stderr = Join-Path $repo "reports\zarr_build_stderr_resume.log"
$stdout = Join-Path $repo "reports\zarr_build_stdout_resume.log"
Set-Location $repo
$attempt = 0
while ($true) {
  $attempt++
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Add-Content $wrapperLog "`n=== RETRY $attempt at $ts ==="
  # append separators into stderr log
  Add-Content $stderr "`n=== RETRY $attempt at $ts ===`n"
  $p = Start-Process -FilePath $exe -ArgumentList @("load", $zarr) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError "$stderr.tmp" -PassThru -WindowStyle Hidden -Wait
  # merge tmp stderr into main log
  if (Test-Path "$stderr.tmp") {
    Get-Content "$stderr.tmp" | Add-Content $stderr
    Remove-Item "$stderr.tmp" -Force -EA SilentlyContinue
  }
  $code = $p.ExitCode
  Add-Content $wrapperLog "exit=$code"
  if ($code -eq 0) {
    Add-Content $wrapperLog "LOAD SUCCEEDED"
    # run statistics
    Add-Content $wrapperLog "Starting statistics..."
    $p2 = Start-Process -FilePath $exe -ArgumentList @("statistics", $zarr) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError "$stderr.tmp" -PassThru -WindowStyle Hidden -Wait
    if (Test-Path "$stderr.tmp") { Get-Content "$stderr.tmp" | Add-Content $stderr; Remove-Item "$stderr.tmp" -Force -EA SilentlyContinue }
    Add-Content $wrapperLog "statistics exit=$($p2.ExitCode)"
    break
  }
  Add-Content $wrapperLog "failed; sleeping 60s then retry"
  Start-Sleep -Seconds 60
}
