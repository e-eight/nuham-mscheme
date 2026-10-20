#!/bin/bash
# One authorized diagnostic job; no production calculation or automatic retry.
#SBATCH --job-name=nuham-zero-diagnostic
#SBATCH --account=bhhk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --constraint=projects
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=512M
#SBATCH --time=00:01:00
#SBATCH --output=zero-diagnostic-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the isolated checkout}"
test "$(git rev-parse HEAD)" = "${1:?Pass the deployed commit SHA}"
test -z "$(git status --porcelain)"
sha256sum --check tools/fixtures/SHA256SUMS
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME
exec /projects/bhhk/soham/nuham-mscheme/venv/bin/python tools/diagnose_zeros3.py \
  --fixture tools/fixtures/zero3-laptop.npz \
  --out "runs/zero-diagnostic-${SLURM_JOB_ID:?}"
