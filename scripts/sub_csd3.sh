#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/rds/user/cs-rami1/rds-t2-cs119/guillem/cache_llm/output/%j.out

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source ~/.bashrc
conda activate cache

cd /home/$USER/cache_llm

export PART=csd3
export DATA_PATH=/rds/user/cs-rami1/rds-t2-cs119/guillem/datasets/data

bash scripts/cluster.sh
export BUDGET=100
export RETRAIN_FREQ=100
bash scripts/run.sh
export BUDGET=2000
export RETRAIN_FREQ=500
bash scripts/run.sh

