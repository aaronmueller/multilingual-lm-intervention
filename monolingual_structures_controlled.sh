#!/bin/bash

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`
source /home/amueller/miniconda3/bin/activate
conda activate xtreme

EN_MODEL="bert-base-cased"

python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL none controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 none controlled language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased none controlled language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base none controlled language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased none controlled language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL prep_singular controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 prep_singular controlled language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased prep_singular controlled language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base prep_singular controlled language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased prep_singular controlled language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL prep_plural controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 prep_plural controlled language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased prep_plural controlled language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base prep_plural controlled language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased prep_plural controlled language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL rc_singular controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 rc_singular controlled language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base rc_singular controlled language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased rc_singular controlled language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased rc_singular controlled language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL rc_plural controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 rc_plural controlled language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base rc_plural controlled language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased rc_plural controlled language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased rc_plural controlled language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL distractor controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL distractor_1 controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL bigram controlled language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL bigram_shuffle controlled language:en && wait
