#!/bin/bash
#SBATCH --job-name=projection_baseline
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --time=1-00:00:00
#SBATCH --mem=256G
#SBATCH --output=%j_bootstrap.o
#SBATCH --error=%j_bootstrap.e
#SBATCH --mail-user=emilyshana.cheng@upf.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --exclude=node0[40,44]


source ~/.bashrc;
conda activate neuro;
cd /home/echeng/encoding-models;


python3 main.py \
        --model $MODEL \
        --seed_layer $LAYER \
        --y_projection $PROJ \
        --n_evecs $N_EVECS \
        --target_x_dim $IPCA_X_DIM \
        --which_layers ipca \
        --device cuda \
