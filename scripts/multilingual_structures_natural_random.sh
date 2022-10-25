#!/bin/bash

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`
source /home/amueller/miniconda3/bin/activate
conda activate xglm

# These could be swapped for monolingual models or mBERT.
EN_MODEL="facebook/xglm-564M"
FR_MODEL="facebook/xglm-564M"
DE_MODEL="facebook/xglm-564M"
FI_MODEL="facebook/xglm-564M"
NL_MODEL="facebook/xglm-564M"

for language in {en,fr,de,nl,fi}; do
    model_var=${language^^}_MODEL
    model_name=${!model_var}
    for structure in {none,prep_singular,prep_plural,rc_singular,rc_plural}; do
        python neuron_experiment_multiple_templates_num_agreement.py $model_name $structure natural language:${language}_short random_weights:true && wait
    done
done