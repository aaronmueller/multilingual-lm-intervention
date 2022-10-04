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
#EN_MODEL="bert-base-multilingual-cased"
#FR_MODEL="bert-base-multilingual-cased"
#DE_MODEL="bert-base-multilingual-cased"
#FI_MODEL="bert-base-multilingual-cased"
#NL_MODEL="bert-base-multilingual-cased"

#python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL none natural language:en_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $FI_MODEL none natural language:fi_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $NL_MODEL none natural language:nl_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $FR_MODEL none natural language:fr_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $DE_MODEL none natural language:de_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL prep_singular natural language:en_short && wait
# python neuron_experiment_multiple_templates_num_agreement.py $FI_MODEL prep_singular natural language:fi_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $NL_MODEL prep_singular natural language:nl_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $FR_MODEL prep_singular natural language:fr_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $DE_MODEL prep_singular natural language:de_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL prep_plural natural language:en_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $FI_MODEL prep_plural natural language:fi_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $NL_MODEL prep_plural natural language:nl_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $FR_MODEL prep_plural natural language:fr_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $DE_MODEL prep_plural natural language:de_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL rc_singular natural language:en_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $FI_MODEL rc_singular natural language:fi_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $FR_MODEL rc_singular natural language:fr_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $NL_MODEL rc_singular natural language:nl_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $DE_MODEL rc_singular natural language:de_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL rc_plural natural language:en_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $FI_MODEL rc_plural natural language:fi_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $FR_MODEL rc_plural natural language:fr_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $NL_MODEL rc_plural natural language:nl_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $DE_MODEL rc_plural natural language:de_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL within_rc_singular natural language:en_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $FI_MODEL within_rc_singular natural language:fi_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $FR_MODEL within_rc_singular natural language:fr_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $NL_MODEL within_rc_singular natural language:nl_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $DE_MODEL within_rc_singular natural language:de_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL within_rc_plural natural language:en_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $FI_MODEL within_rc_plural natural language:fi_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $FR_MODEL within_rc_plural natural language:fr_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $NL_MODEL within_rc_plural natural language:nl_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $DE_MODEL within_rc_plural natural language:de_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL distractor natural language:en_short && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL distractor_1 natural language:en_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL bigram natural language:en_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL bigram_shuffle natural language:en_short && wait
