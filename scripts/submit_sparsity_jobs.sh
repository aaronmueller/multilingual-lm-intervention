#!/bin/bash
shopt -s expand_aliases
. ~/.bashrc

qsub_std neuron_sparsity.sh bert-base-cased 96 bigram en
qsub_std neuron_sparsity.sh bert-base-cased 96 bigram_shuffle en
qsub_std neuron_sparsity.sh bert-base-cased 96 semantic_short en
qsub_std neuron_sparsity.sh bert-base-cased 96 semantic_long en
qsub_std neuron_sparsity.sh bert-base-multilingual-cased 96 bigram en
qsub_std neuron_sparsity.sh bert-base-multilingual-cased 96 bigram_shuffle en
qsub_std neuron_sparsity.sh bert-base-multilingual-cased 96 semantic_short en
qsub_std neuron_sparsity.sh bert-base-multilingual-cased 96 semantic_long en
qsub_std neuron_sparsity.sh facebook/xglm-564M 128 bigram en
qsub_std neuron_sparsity.sh facebook/xglm-564M 128 bigram_shuffle en
qsub_std neuron_sparsity.sh facebook/xglm-564M 128 semantic_short en
qsub_std neuron_sparsity.sh facebook/xglm-564M 128 semantic_long en
qsub_std neuron_sparsity.sh gpt2-medium 128 bigram en
qsub_std neuron_sparsity.sh gpt2-medium 128 bigram_shuffle en
qsub_std neuron_sparsity.sh gpt2-medium 128 semantic_short en
qsub_std neuron_sparsity.sh gpt2-medium 128 semantic_long en