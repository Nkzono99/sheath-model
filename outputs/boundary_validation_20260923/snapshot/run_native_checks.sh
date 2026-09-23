#!/bin/bash
set -euo pipefail
cd /home/b/b36291/.cache/runhand/scratch/tasks/analysis/task-f8f98afe1dbd434c891077aad4320b93/work
mkdir -p validation
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export OMP_PROC_BIND=false
export PYTHONPATH="$PWD"
PY=/LARGE0/gr20001/b36291/Github/DRIFT/.venv/bin/python
date -u +%Y-%m-%dT%H:%M:%SZ > validation/native_start.txt
hostname > validation/hostname.txt
module list > validation/modules.txt 2>&1
gfortran --version > validation/compiler.txt
"$PY" -c 'import sys,numpy,scipy; print(sys.version); print("numpy",numpy.__version__,"scipy",scipy.__version__)' > validation/python_environment.txt
fpm test --compiler gfortran --profile release --flag '-ffree-line-length-none' > validation/fortran_tests.log 2>&1
"$PY" -m unittest discover -s tests -v > validation/python_tests.log 2>&1
fpm run --example validation_probe --compiler gfortran --profile release --flag '-ffree-line-length-none' > validation/field_probe.log 2>&1
"$PY" verify_field_roots.py > validation/field_audit.log 2>&1
"$PY" verify_pe_transport.py > validation/pe_transport.log 2>&1
"$PY" verify_upstream.py > validation/upstream.log 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ > validation/native_end.txt
echo 'NATIVE_CHECKS_PASS'
