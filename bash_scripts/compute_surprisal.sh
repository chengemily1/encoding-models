#!/bin/bash
#SBATCH --job-name="surprisal"
#SBATCH -p alien
#SBATCH -t 02-00:00:00
#SBATCH --exclude=node044
#SBATCH --mem=128G
#SBATCH --gres=gpu:0
#SBATCH --qos=alien
#SBATCH -o %j.out
#SBATCH -e %j.err

source  ~/.bashrc;
conda activate neuro;
cd /home/echeng/encoding-models;

python3 /home/echeng/encoding-models/compute_surprisal.py \
    --model $MODEL
