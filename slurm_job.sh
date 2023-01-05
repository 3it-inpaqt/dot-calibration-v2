#!/bin/bash
#SBATCH --account=rrg-rgmelko-ab
#SBATCH --mail-user=victor.yon@usherbrooke.ca
#SBATCH --mail-type=BEGIN,END,FAIL,INVALID_DEPEND,REQUEUE,TIME_LIMIT_90
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=dot-calibration-v2-runs-planner
#SBATCH --output=out/slurm-%j.out
module load geos/3.9.1 python/3.8.10

# Create python virtual env (all package have to be available on the system)
#virtualenv --no-download "$SLURM_TMPDIR"/venv
# source "$SLURM_TMPDIR"/venv/bin/activate
#pip install --no-index --upgrade pip
#pip install -r requirements.txt
#pip install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip

# If the virtual env is already on the working directory
source venv/bin/activate

# Start job
python start_full_exp.py