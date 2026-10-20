#!/bin/bash
#SBATCH --job-name=nuham-w8-repaired
#SBATCH --account=bhhk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --constraint=projects
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --mem=2G
#SBATCH --time=00:04:00
#SBATCH --output=runs/compatibility-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the isolated checkout}"
test "$(git rev-parse HEAD)" = "${1:?Pass the deployed commit SHA}"
test -z "$(git status --porcelain)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME
exec /projects/bhhk/soham/nuham-mscheme/venv/bin/python tools/comparison/compatibility.py \
  --repo . --commit "$1" \
  --source /projects/bhhk/soham/nuham-mscheme/out/N3LO_N3LOEMlnl_hw30to16_e8_e2_16_e3_8.me3j.gz \
  --out "runs/compatibility-${SLURM_JOB_ID:?}"
