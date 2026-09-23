#!/bin/bash
set -euo pipefail
cd /home/b/b36291/.cache/runhand/scratch/tasks/analysis/task-f8f98afe1dbd434c891077aad4320b93/work
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_PROC_BIND=false
export PYTHONPATH="$PWD"
PY=/LARGE0/gr20001/b36291/Github/DRIFT/.venv/bin/python
cp validation/field_audit.json validation/field_audit_endpoints_only.json
"$PY" verify_field_roots.py > validation/field_audit.log 2>&1
"$PY" verify_finite_reservoir.py > validation/finite_reservoir.log 2>&1
"$PY" plot_validation.py > validation/plot.log 2>&1
rg --files -g '*.f90' -g '*.py' src sheath_model test tests example | sort | xargs sha256sum > validation/source_hashes.txt
sha256sum verify_*.py plot_validation.py run_*checks.sh >> validation/source_hashes.txt
date -u +%Y-%m-%dT%H:%M:%SZ > validation/additional_end.txt
echo ADDITIONAL_CHECKS_PASS
