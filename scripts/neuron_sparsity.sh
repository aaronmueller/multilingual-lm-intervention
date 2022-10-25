#!/bin/bash

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`
source /home/amueller/miniconda3/bin/activate
conda activate xglm

model=$1
k=$2
attractor=$3
language=$4

python neuron_intervention_sparsity.py --model_type $model --k $k --algo topkall \
    --attractor $attractor \
    --language  $language