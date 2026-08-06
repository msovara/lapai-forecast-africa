#!/usr/bin/env bash
# Verify the uploaded Track A training Zarr on Lengau matches the laptop source.
# Mirrors the audit used on the abandoned era5-o96-1979-2023 store so a partial
# upload can never be mistaken for a complete one.
Z="${1:-/home/msovara/lustre/lapai-data/era5_n96_2020_2021.zarr}"
LINK="/home/msovara/repos/lapai-forecast/data/processed/lapai/era5_n96_2020_2021.zarr"

echo "store: $Z"
echo

echo "=== symlink from repo path ==="
ls -la "$LINK"
test -d "$LINK" && echo "resolves to directory: YES" || echo "resolves to directory: NO"
echo

echo "=== required anemoi arrays ==="
missing=0
for a in data dates latitudes longitudes mean stdev minimum maximum; do
  if [ -e "$Z/$a" ]; then
    echo "  $a : PRESENT"
  else
    echo "  $a : MISSING"
    missing=$((missing + 1))
  fi
done
echo

echo "=== root metadata ==="
for m in .zgroup .zattrs .zmetadata; do
  test -e "$Z/$m" && echo "  $m : PRESENT" || echo "  $m : absent"
done
echo

echo "=== data chunks ==="
n=$(ls -1 "$Z/data" | wc -l)
echo "  files in data/ : $n"
echo "  expected       : 2925  (2924 chunks + .zarray)"
echo

echo "=== data/.zarray shape ==="
cat "$Z/data/.zarray" 2>/dev/null | tr -d ' \n' | sed 's/.*"shape":\[\([^]]*\)\].*/  shape: [\1]/'
echo

echo "=== zero-length files (truncated transfer) ==="
z=$(find "$Z" -type f -size 0 | wc -l)
echo "  zero-length count: $z"
find "$Z" -type f -size 0 | head -10
echo

echo "=== total size ==="
du -sb "$Z" | awk '{printf "  bytes: %s\n", $1}'
echo "  expected bytes: 28262849790  (laptop source)"
echo

echo "=== leftover rsync temp files ==="
find "$Z" -name '.*.[a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9]' -o -name '*.partial' | head -10
echo "  (none listed above = clean)"
echo

if [ "$missing" -eq 0 ] && [ "$z" -eq 0 ]; then
  echo "VERDICT: store looks complete"
else
  echo "VERDICT: PROBLEM -- missing arrays=$missing zero-length=$z"
fi
