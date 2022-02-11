#!/bin/bash

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`
source /home/amueller/miniconda3/bin/activate
conda activate xtreme

EN_MODEL="bert-base-cased"

python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL none natural language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 none natural language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased none natural language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base none natural language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased none natural language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL prep_singular natural language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 prep_singular natural language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased prep_singular natural language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base prep_singular natural language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased prep_singular natural language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL prep_plural natural language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 prep_plural natural language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased prep_plural natural language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base prep_plural natural language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased prep_plural natural language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL rc_singular natural language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 rc_singular natural language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base rc_singular natural language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased rc_singular natural language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased rc_singular natural language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL rc_plural natural language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 rc_plural natural language:fi && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base rc_plural natural language:fr && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased rc_plural natural language:nl && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased rc_plural natural language:de && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL distractor natural language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL distractor_1 natural language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL bigram natural language:en && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL bigram_shuffle natural language:en && wait
