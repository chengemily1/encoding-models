#!/bin/bash
#SBATCH --job-name=finetune_id
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --output=/home/echeng/%j_bootstrap.o
#SBATCH --error=/home/echeng/%j_bootstrap.e
#SBATCH --mail-user=emilyshana.cheng@upf.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --exclude=node0[40,44]


source ~/.bashrc;
conda activate neuro;
cd /home/echeng/encoding-models;


python3 finetuning_id_computation.py \
        --experiment $EXPERIMENT
