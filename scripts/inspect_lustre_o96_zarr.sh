#!/bin/bash
# Inspect the partial era5-o96-1979-2023-6h-v8.zarr on Lengau lustre.
# Determines whether the `data` array is complete enough to repair into a
# usable anemoi dataset, instead of uploading the 26 GB 2020-2021 store.
Z="${1:-/home/msovara/lustre/era5-o96-1979-2023-6h-v8.zarr}"

echo "store: $Z"
echo

echo "=== top-level entries ==="
ls -1 "$Z"
echo

echo "=== data chunk count ==="
n=$(ls -1 "$Z/data" | wc -l)
echo "files in data/ : $n"
echo "expected       : 65745  (65744 chunks + .zarray)"
echo "missing        : $(( 65745 - n ))"
echo

echo "=== first / last chunk names ==="
ls -1 "$Z/data" | sort -t. -k1,1n | head -3
echo "..."
ls -1 "$Z/data" | sort -t. -k1,1n | tail -3
echo

echo "=== highest chunk index present ==="
ls -1 "$Z/data" | grep -E '^[0-9]+\.' | cut -d. -f1 | sort -n | tail -1
echo

echo "=== zero-length / truncated chunks ==="
find "$Z/data" -maxdepth 1 -type f -size 0 | head -20
echo "zero-length count: $(find "$Z/data" -maxdepth 1 -type f -size 0 | wc -l)"
echo

echo "=== sibling arrays ==="
for a in maximum minimum squares mean stdev dates latitudes longitudes; do
  if [ -e "$Z/$a" ]; then
    echo "$a : PRESENT ($(ls -1 "$Z/$a" | wc -l) files)"
  else
    echo "$a : MISSING"
  fi
done
echo

echo "=== build state / provenance ==="
ls -la "$Z" | head -20
find "$Z" -maxdepth 1 -name '*.json' -o -maxdepth 1 -name '.z*' | head
