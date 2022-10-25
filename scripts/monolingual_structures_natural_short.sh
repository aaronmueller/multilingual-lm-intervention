#!/bin/bash

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`
source /home/amueller/miniconda3/bin/activate
conda activate xglm

EN_MODEL="bert-base-cased"
FR_MODEL="camembert-base"
DE_MODEL="bert-base-german-cased"
NL_MODEL="GroNLP/bert-base-dutch-cased"
FI_MODEL="TurkuNLP/bert-base-finnish-cased-v1"

for language in {en,fr,de,nl,fi}; do
    model_var=${language^^}_MODEL
    model_name=${!model_var}
    for structure in {none,prep_singular,prep_plural,rc_singular,rc_plural}; do
        python neuron_experiment_multiple_templates_num_agreement.py $model_name $structure natural language:${language}_short && wait
    done
done