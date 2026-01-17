#!/bin/bash
#SBATCH --job-name=projection
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=256G
#SBATCH --output=/scratch/colt/echeng/cache/%j_bootstrap.o
#SBATCH --error=/scratch/colt/echeng/cache/%j_bootstrap.e
#SBATCH --mail-user=emilyshana.cheng@upf.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --exclude=node0[40,41,44]


source ~/.bashrc;
conda activate neuro;
cd /home/echeng/encoding-models;


python3 main.py \
        --model $MODEL \
        --seed_layer $LAYER \
        --y_projection $PROJ \
        --subject $SUBJECT \
        --n_evecs $N_EVECS \
        --which_layers single \
        --device cuda \
