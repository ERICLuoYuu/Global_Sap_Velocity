#!/bin/bash
#SBATCH --job-name=static_module
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=8G --time=02:00:00
#SBATCH --output=/scratch/tmp/yluo2/static_logs/%x_%j.out
#SBATCH --error=/scratch/tmp/yluo2/static_logs/%x_%j.err

# Runs one module of the package from the worktree: job_module.sh <module> [args]. Run by -m with
# the worktree on PYTHONPATH, because a script run by path from the main clone would silently
# import the main clone's src. Resources are set on the sbatch line. Exit code is the module's own.
set -uo pipefail
hostname; date
W=/scratch/tmp/yluo2/gsv-wt/static-features
cd "$W" || exit 1
source /scratch/tmp/yluo2/gsv/.venv/bin/activate || exit 1
export PYTHONPATH="$W"
export PYTHONHASHSEED=0
python --version

MODULE="$1"; shift
python -u -W ignore::FutureWarning -m "$MODULE" "$@"
RC=$?
echo "### module exit: $RC"
exit $RC
