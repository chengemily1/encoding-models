#!/bin/bash

for model in whisper #wavlm EleutherAI/pythia-160m facebook/opt-125m facebook/opt-1.3b EleutherAI/pythia-6.9b-deduped facebook/opt-13b 
do
    export MODEL=$model;
    export LAYER=1; # random choice

    for target_x_dim in 250 500 1000 2500 5000
    do
        export IPCA_X_DIM=$target_x_dim;
        model_str=$(basename "$MODEL");
    
        # Save directory
        save_dir="/home/echeng/encoding-models/results/${model_str}"
    
        # Ensure directory exists
        mkdir -p "$save_dir"
    
        # Run PCA-ed y
        for n_evecs in 50 100 200 300 400 1000 2000
        do
            export N_EVECS=$n_evecs;
            export PROJ=pca;
            json_file="${save_dir}/results_ipca_n_layers_1_seed_layer_${LAYER}_y_rank_${N_EVECS}.0_${PROJ}_ridge.json"
    
            # If file exists, skip/continue
            if [ -e "$json_file"  ]; then
                echo "Skipping $json_file (already exists)"
            else
                sbatch --export=ALL ./bash_scripts/all_layer_encode.sh;
            fi
        done

        # Run I baseline
        export PROJ=I;
        export N_EVECS=1000;
    
        sbatch --export=ALL ./bash_scripts/all_layer_encode.sh;
        
    done
done
