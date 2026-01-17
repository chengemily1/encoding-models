#!/bin/bash

for exp in pretrained
do
    export EXPERIMENT=$exp;
    sbatch --export=ALL ./bash_scripts/finetuning_id.sh;
done