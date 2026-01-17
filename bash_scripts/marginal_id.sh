#!/bin/bash
#SBATCH --job-name=marginal_id
#SBATCH --partition=high
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --output=/home/echeng/%j_bootstrap.o
#SBATCH --error=/home/echeng/%j_bootstrap.e
#SBATCH --mail-user=emilyshana.cheng@upf.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --exclude=node0[40,44]


source ~/.bashrc;
conda activate neuro;
cd /home/echeng/encoding-models;


python3 marginal_id.py \
        --model $MODEL \
        --subject $SUBJECT \
        --add_delays $DELAY
