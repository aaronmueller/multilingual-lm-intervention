#!/bin/bash

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`
source /home/amueller/miniconda3/bin/activate
conda activate xtreme

MODEL_EN="bert-base-cased"
MODEL_FR="camembert-base"

#python neuron_experiment_multiple_templates_num_agreement.py $MODEL_EN cuda . false none 3 200 short && wait
#python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 cuda . false none 3 200 fi && wait
#python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased cuda . false none 3 200 nl && wait
python neuron_experiment_multiple_templates_num_agreement.py $MODEL_FR cuda . false none 3 200 fr_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $MODEL_EN cuda . false prep_singular 3 200 short && wait
#python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 cuda . false prep_singular 3 200 fi && wait
#python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased cuda . false prep_singular 3 200 nl && wait
python neuron_experiment_multiple_templates_num_agreement.py $MODEL_FR cuda . false prep_singular 3 200 fr_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $MODEL_EN cuda . false prep_plural 3 200 short && wait
#python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 cuda . false prep_plural 3 200 fi && wait
#python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased cuda . false prep_plural 3 200 nl && wait
python neuron_experiment_multiple_templates_num_agreement.py $MODEL_FR cuda . false prep_plural 3 200 fr_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $MODEL_EN cuda . false rc_singular 3 200 short && wait
#python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 cuda . false rc_singular 3 200 fi && wait
python neuron_experiment_multiple_templates_num_agreement.py $MODEL_FR cuda . false rc_singular 3 200 fr_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased cuda . false rc_singular 3 200 nl && wait
#python neuron_experiment_multiple_templates_num_agreement.py $MODEL_EN cuda . false rc_plural 3 200 short && wait
#python neuron_experiment_multiple_templates_num_agreement.py bert-base-finnish-cased-v1 cuda . false rc_plural 3 200 fi && wait
python neuron_experiment_multiple_templates_num_agreement.py $MODEL_FR cuda . false rc_plural 3 200 fr_short && wait
#python neuron_experiment_multiple_templates_num_agreement.py bert-base-dutch-cased cuda . false rc_plural 3 200 nl && wait
#python neuron_experiment_multiple_templates_num_agreement.py $MODEL_EN cuda . false distractor 3 200 short && wait
#python neuron_experiment_multiple_templates_num_agreement.py $MODEL_EN cuda . false distractor_1 3 200 short && wait
# python neuron_experiment_multiple_templates_num_agreement.py bert-base-multilingual-cased cuda . false bigram_shuffle 3 200 && wait
