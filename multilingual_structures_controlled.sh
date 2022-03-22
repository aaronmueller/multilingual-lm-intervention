#!/bin/bash

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`
source /home/amueller/miniconda3/bin/activate
conda activate xglm

EN_MODEL="facebook/xglm-564M"
FR_MODEL="facebook/xglm-564M"
DE_MODEL="facebook/xglm-564M"
FI_MODEL="facebook/xglm-564M"
NL_MODEL="facebook/xglm-564M"

python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL none controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $FI_MODEL none controlled language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py $NL_MODEL none controlled language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py $FR_MODEL none controlled language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py $DE_MODEL none controlled language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL prep_singular controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $FI_MODEL prep_singular controlled language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py $NL_MODEL prep_singular controlled language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py $FR_MODEL prep_singular controlled language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py $DE_MODEL prep_singular controlled language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL prep_plural controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $FI_MODEL prep_plural controlled language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py $NL_MODEL prep_plural controlled language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py $FR_MODEL prep_plural controlled language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py $DE_MODEL prep_plural controlled language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL rc_singular controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $FI_MODEL rc_singular controlled language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py $FR_MODEL rc_singular controlled language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py $NL_MODEL rc_singular controlled language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py $DE_MODEL rc_singular controlled language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL rc_plural controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $FI_MODEL rc_plural controlled language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py $FR_MODEL rc_plural controlled language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py $NL_MODEL rc_plural controlled language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py $DE_MODEL rc_plural controlled language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL distractor controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL distractor_1 controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL bigram controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL bigram_shuffle controlled language:en && wait
