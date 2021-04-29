#!/bin/bash
#SBATCH --account=rrg-rgmelko-ab
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --job-name=dot-calibration-v2-runs-planner
module load geos python/3.8.2

# Create python virtual env
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt

# Start job
python runs_planner.py