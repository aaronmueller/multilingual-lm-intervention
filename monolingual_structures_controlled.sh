#!/bin/bash

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`
source /home/amueller/miniconda3/bin/activate
conda activate xtreme

EN_MODEL="bert-base-cased"

python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL cuda . false none 3 200 en zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 cuda . false none 3 200 fi zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased cuda . false none 3 200 nl zero && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base cuda . false none 3 200 fr zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased cuda . false none 3 200 de zero && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL cuda . false prep_singular 3 200 en zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 cuda . false prep_singular 3 200 fi zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased cuda . false prep_singular 3 200 nl zero && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base cuda . false prep_singular 3 200 fr zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased cuda . false prep_singular 3 200 de zero && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL cuda . false prep_plural 3 200 en zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 cuda . false prep_plural 3 200 fi zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased cuda . false prep_plural 3 200 nl zero && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base cuda . false prep_plural 3 200 fr zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased cuda . false prep_plural 3 200 de zero && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL cuda . false rc_singular 3 200 en zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 cuda . false rc_singular 3 200 fi zero && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base cuda . false rc_singular 3 200 fr zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased cuda . false rc_singular 3 200 nl zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased cuda . false rc_singular 3 200 de zero && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL cuda . false rc_plural 3 200 en zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 cuda . false rc_plural 3 200 fi zero && wait
python neuron_experiment_multiple_templates_num_agreement.py camembert-base cuda . false rc_plural 3 200 fr zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased cuda . false rc_plural 3 200 nl zero && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased cuda . false rc_plural 3 200 de zero && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL cuda . false distractor 3 200 en zero && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL cuda . false distractor_1 3 200 en zero && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL cuda . false bigram 3 200 en zero && wait
python neuron_experiment_multiple_templates_num_agreement.py $EN_MODEL cuda . false bigram_shuffle 3 200 en zero && wait
