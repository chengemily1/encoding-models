#!/bin/bash

for layer in {1..24}
do
    export LAYER=$layer;
    
    for subject in 2 3
    do
        for model in EleutherAI/pythia-410m-seed2 EleutherAI/pythia-410m-seed3 EleutherAI/pythia-410m-seed4 EleutherAI/pythia-410m-seed5 EleutherAI/pythia-410m-seed6 EleutherAI/pythia-410m-seed7 EleutherAI/pythia-410m-seed8 EleutherAI/pythia-410m-seed9 #EleutherAI/pythia-6.9b-deduped # whisper facebook/opt-125m EleutherAI/pythia-160m  facebook/opt-1.3b
        do
            # Run baseline
            export PROJ=I;
            export N_EVECS=1000;
            export SUBJECT=$subject
            export MODEL=$model;

            # Extract model_str from model (like Python's split('/')[-1])
            model_str=$(basename "$MODEL")

            # Save directory
            save_dir="/home/echeng/encoding-models/results/${model_str}/UTS0${SUBJECT}"

            # Ensure directory exists
            mkdir -p "$save_dir"
            json_file="${save_dir}/results_single_n_layers_1_seed_layer_${LAYER}_x_rank_1000_y_rank_${N_EVECS}.0_${PROJ}_ridge.json"
            # If file exists, skip/continue
            if [ -e "$json_file"  ]; then
                echo "Skipping $json_file (already exists)"
            else
                sbatch --export=ALL ./bash_scripts/single_layer_encode.sh;
            fi

            # Run PCA
            # for n_evecs in 200 #300 #400 1000 2000
            # do
            #     export N_EVECS=$n_evecs;
            #     export PROJ=pca;
            #     json_file="${save_dir}/results_single_n_layers_1_seed_layer_${LAYER}_y_rank_${N_EVECS}.0_${PROJ}_ridge.json"

            #     # If file exists, skip/continue
            #     if [ -e "$json_file"  ]; then
            #         echo "Skipping $json_file (already exists)"
            #     else
            #         sbatch --export=ALL ./bash_scripts/single_layer_encode.sh;
            #     fi
            # done
        done
    done
done
