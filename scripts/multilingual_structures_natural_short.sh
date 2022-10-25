#!/bin/bash

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`
source /home/amueller/miniconda3/bin/activate
conda activate xglm

# We use either `bert-base-multilingual-cased` or `xglm-564M`
MULTI_MODEL=$1

for language in {en,fr,de,nl,fi}; do
    model_var=${language^^}_MODEL
    model_name=${!model_var}
    for structure in {none,prep_singular,prep_plural,rc_singular,rc_plural}; do
        python neuron_experiment_multiple_templates_num_agreement.py $model_name $structure natural language:${language}_short && wait
    done
done