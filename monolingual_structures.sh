#!/bin/bash

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`
source /home/amueller/miniconda3/bin/activate
conda activate xtreme

#python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 cuda . false none 3 200 fi && wait
#python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased cuda . false none 3 200 nl && wait
#python neuron_experiment_multiple_templates_num_agreement.py camembert-base cuda . false none 3 200 fr && wait
# python neuron_experiment_multiple_templates_num_agreement.py bert-base-cased cuda . false rc_singular 3 200 && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 cuda . false rc_singular 3 200 fi && wait
#python neuron_experiment_multiple_templates_num_agreement.py camembert-base cuda . false rc_singular 3 200 fr && wait
#python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased cuda . false rc_singular 3 200 nl && wait
python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 cuda . false rc_plural 3 200 fi && wait
#python neuron_experiment_multiple_templates_num_agreement.py camembert-base cuda . false rc_plural 3 200 fr && wait
#python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased cuda . false rc_plural 3 200 nl && wait
# python neuron_experiment_multiple_templates_num_agreement.py bert-base-cased cuda . false distractor 3 200
# python neuron_experiment_multiple_templates_num_agreement.py bert-base-cased cuda . false distractor_1 3 200
# python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased cuda . false bigram_shuffle 3 200 && wait
