#!/bin/bash

for model in facebook/opt-13b EleutherAI/pythia-6.9b-deduped EleutherAI/pythia-410m-seed2 EleutherAI/pythia-410m-seed3 EleutherAI/pythia-410m-seed4 EleutherAI/pythia-410m-seed8 EleutherAI/pythia-410m-seed9
do
    export MODEL=$model;
    sbatch --export=ALL ./bash_scripts/compute_surprisal.sh
done
