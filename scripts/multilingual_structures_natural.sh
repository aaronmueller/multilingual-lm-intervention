#!/bin/bash

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`
source /home/amueller/miniconda3/bin/activate
conda activate xglm

# use `bert-base-multilingual-cased` or `xglm-564M`
MULTI_MODEL=$1

for language in {en,fr,de,nl,fi}; do
    for structure in {none,prep_singular,prep_plural,rc_singular,rc_plural}; do
        python neuron_experiment_multiple_templates_num_agreement.py $MULTI_MODEL $structure natural language:${language} && wait
    done
done

# Simple agreement, randomized baseline
python neuron_experiment_multiple_templates_num_agreement.py $MULTI_MODEL none natural language:en random_weights:true && wait

# Semantic plausibility/Token collocation baselines
python neuron_experiment_multiple_templates_num_agreement.py $MULTI_MODEL bigram natural language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $MULTI_MODEL bigram_shuffle natural language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $MULTI_MODEL semantic_short natural language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $MULTI_MODEL semantic_long natural language:en && wait