#!/bin/bash
set -uo pipefail
L=/home/msovara/lustre/dev/lapai-anemoi/bin/python
echo "=== installing scikit-learn 1.7.2 (glibc 2.17) into lustre env ==="
"$L" -m pip install --no-index --find-links "$HOME/wheelhouse_sklearn" --force-reinstall --no-deps "scikit-learn==1.7.2" 2>&1 | tail -8
echo "=== verify sklearn version ==="
"$L" -c 'import sklearn; print("sklearn", sklearn.__version__)'
echo "=== ball_tree glibc requirement ==="
SO=$("$L" -c 'import sklearn,glob,os; print(glob.glob(os.path.join(os.path.dirname(sklearn.__file__),"neighbors","_ball_tree*.so"))[0])' 2>/dev/null)
objdump -T "$SO" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -V | uniq | tail -3
echo "=== anemoi.graphs import ==="
"$L" -c 'from anemoi.graphs.nodes import AnemoiDatasetNodes; print("anemoi.graphs import OK")'
