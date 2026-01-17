#!/bin/bash

for add_delay in 0 
do
    export DELAY=$add_delay;
    for model in whisper wavlm facebook/opt-13b EleutherAI/pythia-160m facebook/opt-125m facebook/opt-1.3b EleutherAI/pythia-6.9b-deduped
    do
        export MODEL=$model;

        for subj in 2 3
        do
            export SUBJECT=$subj;
            sbatch --export=ALL ./bash_scripts/marginal_id.sh;
        done
    done
done
