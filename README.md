# Causal Analysis of Syntactic Agreement Neurons in Multilingual Language Models

This repository contains data and code for replicating the results of our CoNLL 2022 paper *Causal Analysis of Syntactic Agreement Neurons in Multilingual Language Models*.

We include code for running neuron intervention experiments. Our code is essentially a multilingual extension of the repository for [Finlayson et al. (2021)](https://github.com/mattf1n/lm-intervention); we augment the dataset to include stimuli from French, German, Dutch, and Finnish. We also augment the code to be compatible with structures in multiple languages. We also implement support for controlled indirect effect experiments, though in our paper, we only use natural indirect effects.

Our code is written primarily for use with (Ro)BERT(a)-based models (including GermanBERT, BERTje, FinnishBERT, and CamemBERT) as well as XGLM, though as we use [huggingface transformers](https://github.com/huggingface/transformers/), this code should be extensible to other masked and autoregressive language models as well.

## Requirements

`torch==1.8.2`
`transformers==4.18`

## Data Templates

Templates are translated and slightly modified from the stimuli of [Finlayson et al. (2021)](https://github.com/mattf1n/lm-intervention), which are based on those of [Lakretz et al. (2019)](https://github.com/FAIRNS/Number_and_syntax_units_in_LSTM_LMs).

The structures are as follows:

| Structure | Template |
| --- | --- |
| Simple | The [noun] [verb] |
| Across prepositional phrase | The [noun] [prep] the [prepnoun] [verb] |
| Across relative clause | The [noun] the [noun2] [verb2] [verb] |

Our semantic plausibility baselines and token collocation baselines are structured as follows:

| Structure | Template |
| --- | --- |
| Semantic plausibility (adjacent) | The [adj] [noun] |
| Semantic plausibility (non-adjacent) | The [noun] is [adj] |
| Bigram | [adj] [noun] |

Terminal values for [noun], [verb], [prepnoun], etc., can be found in `vocab/wordlists`.

## Experiments

### Neuron interventions

To obtain neuron indirect effect and total effect numbers, run 
```
python neuron_experiment_multiple_templates_num_agreement.py \
<model> <structure> <intervention_type> (language:<>, out_dir:<> random_weights:<> examples:<>)
```
from the root directory of this repository. We describe the arguments to this script here:

```
model: The name of or path to a huggingface model.
  bert-base-cased | bert-base-multilingual-cased | bert-base-german-cased |
  bert-base-finnish-cased-v1 | bert-base-dutch-cased |
  xglm-564M | gpt2

structure: Which syntactic structure to use.
  none | prep_singular | prep_plural | rc_singular | rc_plural |
  bigram | bigram_shuffle | semantic_short | semantic_long

intervention_type: Whether to use natural or controlled indirect effects. We only use `natural` in our paper.
  natural | controlled

device: Device for pytorch
  cpu | cuda

out_dir: Output directory for the results.

random_weights: Whether to use a randomly initialized (untrained) model.
  true | false

examples: Integer number of examples to use from the template. 
  0 uses all examples. We use 200 in our paper.
```

We provide scripts with example commands in the `scripts` directory.

Running this experiment can take a few hours on a GPU. The outputs also require gigabytes of space for the largest models. This is why we sample a few hundred examples, rather than run on all possible sentences that could be generated from our vocabulary.

Results are output to `results/<date>_neuron_intervention/<structure>_<language>_<direct|indirect>_<model>_<natural|controlled>.csv`
Once all outputs are obtained, running `python make_feathers.py <out_dir>/results` generates `.feather` files from the `.csv` files. We then move the `.feather` files to the `results/feathers/` folder.

The `plots.ipynb` notebook assumes that results will be in this `.feather` format.

## Generating figures

The majority of figures from the paper are generated in `results/feathers/plots.ipynb`. This includes indirect effect contour graphs and neuron overlap graphs.

To generate neuron sparsity graphs,

## Reference

If you use the resources in this repository, please cite our paper:

```
@inproceedings{finlayson-2021-causal,
    title = "Causal Analysis of Syntactic Agreement Neurons in Multilingual Language Models",
    author = "Mueller, Aaron and
    Xia, Yu and
    Linzen, Tal",
    booktitle = "Proceedings of the 2022 Conference on Computational Natural Language Learning (CoNLL)",
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics"
```

## Other Papers / Resources

Code is based on the repository for [Vig et al. (2020)](https://github.com/sebastianGehrmann/CausalMediationAnalysis)

Syntactic agreement templates are based on [Finlayson et al. (2021)](https://github.com/mattf1n/lm-intervention), which are based on materials from [Lakretz et al. (2019)](https://github.com/FAIRNS/Number_and_syntax_units_in_LSTM_LMs)