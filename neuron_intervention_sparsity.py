# -*- coding: utf-8 -*-

import gc
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from transformers import (
    GPT2Tokenizer, BertTokenizer, TransfoXLTokenizer,
    CamembertTokenizer, XLMRobertaTokenizer, XLNetTokenizer,
    XGLMTokenizer
)

from experiment_num_agreement import Intervention, Model
from utils_num_agreement import convert_results_to_pd
from vocab_utils import get_nouns, get_nouns2, get_verbs, get_verbs2, get_prepositions, \
        get_preposition_nouns, get_adv1s, get_adv2s
from generate_all_sentences import load_nouns, load_verbs, load_nouns2, load_verbs2, load_bigrams, \
        load_adv1, load_adv2, load_prepositions, load_preposition_nouns
from neuron_experiment_multiple_templates_num_agreement import construct_interventions_fr, \
        construct_templates_fr, construct_interventions_bi
import vocab_utils as vocab

np.random.seed(1)
torch.manual_seed(1)

def get_template_list():
    # Get list of all considered templates
    # "That" sentences are ours
    # "Because" sentences are a subset
    # from https://arxiv.org/pdf/1807.11714.pdf (Lu et al.)
    return ["The {} said that",
            "The {} yelled that",
            "The {} whispered that",
            "The {} wanted that",
            "The {} desired that",
            "The {} wished that",
            "The {} ate because",
            "The {} ran because",
            "The {} drove because",
            "The {} slept because",
            "The {} cried because",
            "The {} laughed because",
            "The {} went home because",
            "The {} stayed up because",
            "The {} was fired because",
            "The {} was promoted because",
            "The {} yelled because"]

def get_intervention_types():
    return ['man_indirect',
            'woman_indirect']

def construct_templates(attractor):
    # specify format of inputs. fill in with terminals later
    templates = []
    if attractor in  ['prep_singular', 'prep_plural']:
        for p in get_prepositions():
            for ppns, ppnp in get_preposition_nouns():
                ppn = ppns if attractor == 'prep_singular' else ppnp
                template = ' '.join(['The', '{}', p, 'the', ppn])
                templates.append(template)
    elif attractor in ('rc_singular', 'rc_plural', 'rc_singular_no_that', 'rc_plural_no_that'):
        for noun2s, noun2p in get_nouns2():
            noun2 = noun2s if attractor.startswith('rc_singular') else noun2p
            for verb2s, verb2p in get_verbs2():
                verb2 = verb2s if attractor.startswith('rc_singular') else verb2p
                if attractor.endswith('no_that'):
                    template = ' '.join(['The', '{}', 'the', noun2, verb2])
                else:
                    template = ' '.join(['The', '{}', 'that', 'the', noun2, verb2])
                templates.append(template)
    elif attractor in ('within_rc_singular', 'within_rc_plural', 'within_rc_singular_no_that', 'within_rc_plural_no_that'):
        for ns, np in vocab.get_nouns():
            noun = ns if attractor.startswith('within_rc_singular') else np
            if attractor.endswith('no_that'):
                template = ' '.join(['The', noun, 'the', '{}'])
            else:
                template = ' '.join(['The', noun, 'that', 'the', '{}'])
            templates.append(template)
    elif attractor == 'distractor':
        for  adv1 in  get_adv1s():
            for adv2 in get_adv2s():
                templates.append(' '.join(['The', '{}', adv1, 'and', adv2]))
    elif attractor == 'distractor_1':
        for adv1 in get_adv1s():
            templates.append(' '.join(['The', '{}', adv1]))

    else:   # defaults to simple agreement
        templates = ['The {}']
    return templates

def construct_templates_bi(attractor):
    if attractor == "semantic":
        attractor = "semantic_short"

    word1_list, word2_list = load_bigrams(attractor)

    if attractor.startswith("bigram"):
        temp = ["{}"]
    elif attractor == "semantic_short":
        temp = ["The {}"]     # the <adj> <noun>
        word1_list, word2_list = word2_list, word1_list
    elif attractor == "semantic_long":
        temp = ["The {} is"]  # the <noun> is <adj>
    return temp

def construct_interventions(tokenizer, DEVICE, attractor, seed, examples, intervention_method = "natural"):
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    templates = construct_templates(attractor)
    for temp in templates:
        if attractor.startswith('within_rc'):
            for noun2s, noun2p in get_nouns2():
                if intervention_method == "controlled":
                    noun_list = [noun2s]
                else:
                    noun_list = [noun2s, noun2p]
                for v_singular, v_plural in vocab.get_verbs():
                    all_word_count += 1
                    try:
                        intervention_name = '_'.join([temp, noun2s, v_singular])
                        interventions[intervention_name] = Intervention(
                            tokenizer,
                            temp,
                            noun_list,
                            [v_singular, v_plural],
                            device=DEVICE,
                            method = intervention_method)
                        used_word_count += 1
                    except Exception as e:
                        pass
        else:
            for ns, np in vocab.get_nouns():
                if intervention_method == "controlled":
                    noun_list = [ns]
                else:
                    noun_list = [ns, np]
                for v_singular, v_plural in vocab.get_verbs():
                    all_word_count += 1
                    try: 
                        intervention_name = '_'.join([temp, ns, v_singular])
                        interventions[intervention_name] = Intervention(
                            tokenizer,
                            temp,
                            noun_list,
                            [v_singular, v_plural],
                            device=DEVICE,
                            method = intervention_method)
                        used_word_count += 1
                    except Exception as e:
                        pass
    print(f"\t Only used {used_word_count}/{all_word_count} nouns due to tokenizer")
    if examples > 0 and len(interventions) >= examples:     # randomly sample input sentences
        random.seed(seed)
        interventions = {k: v 
                for k, v in random.sample(interventions.items(), examples)}
    return interventions


def compute_odds_ratio(df, col='odds_ratio'):
  # filter some stuff out
  #df = df[df['candidate2_base_prob'] > 0.01]
  #df = df[df['candidate1_base_prob'] > 0.01]

  odds_base = df['candidate2_base_prob'] / df['candidate1_base_prob']
  odds_intervention = df['candidate2_prob'] / df['candidate1_prob']

  odds_ratio = odds_intervention / odds_base
  df[col] = odds_ratio
  return df

def sort_odds_obj(df):
  df['odds_diff'] = df['odds_ratio'].apply(lambda x: x-1)

  df_sorted = df.sort_values(by=['odds_diff'], ascending=False)
  return df_sorted

# get global list 
def get_all_contrib(model, templates, tokenizer, attractor, language, out_dir='sparsity'):
  out_path = out_dir + "/marg_contrib_" + attractor + "_" + language + "_" + model_name + ".pickle"
  if os.path.exists(out_path):
    marg_contrib = pickle.load(open(out_path, "rb"))
    return marg_contrib['layer'], marg_contrib['neuron']

  df = get_intervention_results(model, templates, tokenizer, attractor=attractor, language=language)
  gc.collect()

  # compute odds ratio differently for each gender
  df = compute_odds_ratio(df)
  df = df[['layer','neuron', 'odds_ratio']]
  gc.collect()
  # merge and average
  df = df.groupby(['layer','neuron'], as_index=False).mean()
  df_sorted = sort_odds_obj(df)
  layer_list = df_sorted['layer'].values
  neuron_list = df_sorted['neuron'].values
  odds_list = df_sorted['odds_ratio'].values

  marg_contrib = {}
  marg_contrib['layer'] = layer_list
  marg_contrib['neuron'] = neuron_list
  marg_contrib['val'] = odds_list
  
  pickle.dump(marg_contrib, open(out_path, "wb" ))
  return layer_list, neuron_list

def get_intervention_results(model, templates, tokenizer, DEVICE='cuda', attractor='none', language='en',
                             seed=12, examples=200, layers_to_adj=[], neurons_to_adj=[], intervention_loc='all',
                             df_layer=None, df_neuron=None, intervention_method="natural"):
  
  intervention_type = 'indirect'

  df = []
  # for template in templates:
    # pickle.dump(template + "_" + gender, open("results/log.pickle", "wb" ) )
  if language == 'en':
    if 'bigram' in attractor or 'semantic' in attractor:
      shuffle = 'shuffle' in attractor
      interventions = construct_interventions_bi(attractor, tokenizer, DEVICE, seed, examples, shuffle, intervention_method)
    else:
      interventions = construct_interventions(tokenizer, DEVICE, attractor, seed, examples, intervention_method)
  else:
    interventions = construct_interventions_fr(tokenizer, DEVICE, attractor, seed, examples, language, intervention_method)
  intervention_results = model.neuron_intervention_experiment(interventions, intervention_type, 
                                                            layers_to_adj=layers_to_adj, neurons_to_adj=neurons_to_adj,
                                                            intervention_loc=intervention_loc)
  df_template = convert_results_to_pd(interventions, intervention_results, df_layer, df_neuron)
  # calc odds ratio and odds-abs 
  df.append(df_template)
  gc.collect()
  return pd.concat(df)

def get_neuron_intervention_results(model, templates, tokenizer, layers, neurons):
    df = get_intervention_results(model, templates, tokenizer,
                                       layers_to_adj=layers, neurons_to_adj=[neurons], intervention_loc='neuron',
                                        df_layer=layers, df_neuron=neurons[0])
    df = compute_odds_ratio(df)
    return df['odds_ratio'].mean()

def top_k_by_layer(model, model_type, tokenizer, attractor, templates, layer, layer_list, neuron_list, k=50, out_dir='sparsity'):
  layer_2_ind = np.where(layer_list == layer)[0]
  neuron_2 = neuron_list[layer_2_ind]
  
  odd_abs_list = []
  for i in range(k):
    print(i)
    temp_list = list(neuron_2[:i+1])

    neurons = [temp_list]

    # get marginal contrib to empty set
    df = get_intervention_results(model, templates, tokenizer, attractor=attractor, language=language,
                                       layers_to_adj=len(temp_list)*[layer], neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=layer, df_neuron=neurons[0])
    
    gc.collect()

    # compute odds ratio differently for each gender
    df = compute_odds_ratio(df)

    # merge and average
    odd_abs_list.append(df['odds_ratio'].mean()-1)
  
    pickle.dump(odd_abs_list, open(out_dir + "/topk_" + model_name + '_' + str(layer) + ".pickle", "wb" ) )

def top_k(model, model_type, tokenizer, attractor, language, templates, layer_list, neuron_list, k=50, out_dir='sparsity'):
  odd_abs_list = []

  for i in range(k):
    print(i)
    n_list = list(neuron_list[:i+1])
    l_list = list(layer_list[:i+1])

    neurons = [n_list]
    print(attractor)
    # get marginal contrib to empty set
    df = get_intervention_results(model, templates, tokenizer, attractor=attractor, language=language,
                                       layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=l_list, df_neuron=neurons[0])
    # compute odds ratio differently for each gender
    df = compute_odds_ratio(df)
    # print(df)

    # merge and average
    odd_abs_list.append(df['odds_ratio'].mean()-1)
    # print(odd_abs_list)

    pickle.dump(odd_abs_list, open(out_dir + "/topk_" + model_name + ".pickle", "wb" ))

# iteratively sample top-k neurons by NIE, intervene on all simultaneously
def top_k_all_seq(model, model_type, tokenizer, attractor, language, templates, layer_list, neuron_list, k=50, out_dir='sparsity'):
  odd_abs_list = []
  n_neurons_list = []
  layer_list = [l-1 for l in layer_list]
  
  for i in range(len(neuron_list) // k):
    n_neurons = k * (i+1)
    print(f"Num neurons: {n_neurons}/{len(neuron_list)}")
    n_list = list(neuron_list[:n_neurons+1])
    l_list = list(layer_list[:n_neurons+1])

    neurons = [n_list]
    # get marginal contrib to empty set
    df = get_intervention_results(model, templates, tokenizer, attractor=attractor, language=language,
                                       layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=l_list, df_neuron=neurons[0])
    # compute odds ratio differently for each gender
    df = compute_odds_ratio(df)
    # print(df)

    # merge and average
    n_neurons_list.append(n_neurons)
    odd_abs_list.append(df['odds_ratio'].mean()-1)
    # print(odd_abs_list)
    
    out_dict = {}
    out_dict['layer'] = l_list
    out_dict['neuron'] = n_list
    out_dict['n_neurons'] = n_neurons_list
    out_dict['odds_ratio'] = odd_abs_list
    
  pickle.dump(out_dict, open(out_dir + "/topk_all_seq_" + model_name + "_" + attractor + "_" + language + ".pickle", "wb"))

def greedy_by_layer(model, model_type, tokenizer, attractor, templates, layer, k=50, out_dir='sparsity'):
  neurons = []
  odd_abs_list = []
  neurons = []

  for i in range(k):


    # get marginal contrib to empty set
    df = get_intervention_results(model, templates, tokenizer, attractor=attractor, language=language,
                                       layers_to_adj=layer, neurons_to_adj=neurons, intervention_loc='layer',
                                        df_layer=layer, df_neuron=None)

    # compute odds ratio differently for each gender
    df = compute_odds_ratio(df)
    gc.collect()

    # merge and average
    df = df.groupby(['layer', 'neuron'], as_index=False).mean()
    df_sorted = sort_odds_obj(df)

    neurons.append(df_sorted.head(1)['neuron'].values[0])
    odd_abs_list.append(df_sorted['odds_diff'].values[0])

    greedy_res = {}
    greedy_res['neuron'] = neurons
    greedy_res['val'] = odd_abs_list

    pickle.dump(greedy_res, open(out_dir + "/greedy_" + model_name + "_" + str(layer) + ".pickle", "wb" ))

def greedy(model, model_type, tokenizer, attractor, templates, k=50, out_dir='sparsity'):
  neurons = []
  odd_abs_list = []
  layers = []

  greedy_filename = out_dir + "/greedy_" + model_name + ".pickle"

  if os.path.exists(greedy_filename):
    print('loading precomputed greedy values')
    res = pickle.load( open(greedy_filename, "rb" )) 
    odd_abs_list = res['val']
    layers = res['layer'] 
    neurons = res['neuron']
    k = k - len(odd_abs_list)
  else:
    neurons = []
    odd_abs_list = []
    layers = []

  for i in range(k):
    print(i)

    # get marginal contrib to empty set
    df = get_intervention_results(model, templates, tokenizer, attractor=attractor, language=language,
                                       layers_to_adj=layers, neurons_to_adj=neurons, intervention_loc='all',
                                        df_layer=None, df_neuron=None)

    # compute odds ratio differently for each gender
    df = compute_odds_ratio(df)
    gc.collect()

    # merge and average
    df = df.groupby(['layer', 'neuron'], as_index=False).mean()
    df_sorted = sort_odds_obj(df)

    neurons.append(df_sorted.head(1)['neuron'].values[0])
    layers.append(df_sorted.head(1)['layer'].values[0])
    odd_abs_list.append(df_sorted['odds_diff'].values[0])

    # memory issue
    del df
    gc.collect()

    greedy_res = {}
    greedy_res['layer'] = layers
    greedy_res['neuron'] = neurons
    greedy_res['val'] = odd_abs_list

    pickle.dump(greedy_res, open(greedy_filename, "wb" ))


def random_greedy_by_layer(layer, attractor, k=50, out_dir='sparsity'):
  neurons = []
  odd_abs_list = []
  neurons = []
  el_list = list(range(1,k+1))
  df = []
  for i in range(k):
    

    # get marginal contrib to empty set
    df = get_intervention_results(model, templates, tokenizer, attractor=attractor, language=language,
                                       layers_to_adj=layer, neurons_to_adj=neurons, intervention_loc='layer',
                                        df_layer=layer, df_neuron=None)

    # compute odds ratio differently for each gender
    df = compute_odds_ratio(df)

    # merge and average
    df = df.groupby(['layer', 'neuron'], as_index=False).mean()
    df_sorted = sort_odds_obj(df)

    j = random.choice(el_list)
    neurons.append(df_sorted.head(j)['neuron'].values[-1])
    odd_abs_list.append(df_sorted.head(j)['odds_abs'].values[-1])

  pickle.dump(odd_abs_list, open("rand_greedy_" + str(layer) + ".pickle", "wb" ))
  pickle.dump(neurons, open("rand_greedy_neurons_" + str(layer) + ".pickle", "wb" ))

def test():
  layer_obj = []
  for layer in range(12):
    print(layer)
    neurons = [list(range(768))]
    # get marginal contrib to empty set
    df = get_intervention_results(model, templates, tokenizer, attractor=attractor, language=language,
                                       layers_to_adj=768*[layer], neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=layer, df_neuron=neurons[0])

    # compute odds ratio differently for each gender
    df = compute_odds_ratio(df)

    # merge and average
    print(layer)
    # print(df_sorted['odds_abs'].values[0])
    layer_obj.append(abs(df['odds_ratio'].mean()-1))

  neurons = [12*list(range(768))]
  # get marginal contrib to empty set
  layer_list = []
  for l in range(12):
    layer_list += (768 * [l])
  df = get_intervention_results(model, templates, tokenizer, 
                                     layers_to_adj=layer_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                      df_layer=layer, df_neuron=neurons[0])

  # compute odds ratio differently for each gender
  df = compute_odds_ratio(df)

  # merge and average
  print(layer_obj)
  print('all')
  print(abs(df['odds_ratio'].mean()-1))

if __name__ == '__main__':
    ap = ArgumentParser(description="Neuron subset selection.")
    ap.add_argument('--model_type', type=str, default='bert-base-multilingual-cased')
    ap.add_argument('--algo', type=str, choices=['topk', 'greedy', 'random_greedy', 'test', 'topkall'], default='topk')
    ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--layer', type=int, default=-1)
    ap.add_argument('--out_dir', type=str, default='sparsity')
    ap.add_argument('--attractor', type=str, default='none')
    ap.add_argument('--random_weights', action='store_true')
    ap.add_argument('--language', type=str, default='en')

    args = ap.parse_args()
    
    algo = args.algo
    k = args.k
    layer = args.layer
    out_dir = args.out_dir
    model_type = args.model_type
    model_name = model_type
    if "/" in model_type:
        model_name = model_name.split("/")[-1]
    attractor = args.attractor
    language = args.language
    random_weights = args.random_weights
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model = Model(device='cuda', gpt2_version=model_type, 
        random_weights=random_weights)
    tokenizer = (GPT2Tokenizer if model.is_gpt2 else
                 TransfoXLTokenizer if model.is_txl else
                 XLNetTokenizer if model.is_xlnet else
                 BertTokenizer if model.is_bert else
                 XLMRobertaTokenizer if model.is_xlmr else
                 CamembertTokenizer if model.is_camembert else
                 XGLMTokenizer).from_pretrained(model_type)
    DEVICE = 'cuda'

    if language == 'en':
      if 'bigram' in attractor or 'semantic' in attractor:
        templates = construct_templates_bi(attractor)
      else:
        templates = construct_templates(attractor)
    else:
      templates = construct_templates_fr(language, attractor)

    if args.algo.startswith('topk'):
        marg_contrib_path = out_dir + "/marg_contrib.pickle"
        #if os.path.exists(marg_contrib_path):
        #    print('Using cached marginal contribution')
        #    marg_contrib = pickle.load( open(marg_contrib_path, "rb" )) 
        #    layer_list = marg_contrib['layer']
        #    neuron_list = marg_contrib['neuron']
        #else:
        print('Computing marginal contribution')
        layer_list, neuron_list = get_all_contrib(model, templates, tokenizer, attractor, language, out_dir)
        if args.algo == 'topk':
            if layer == -1:
                top_k(model, model_type, tokenizer, attractor, language, templates, layer_list, neuron_list, k, out_dir)
            elif layer != -1:
                top_k_by_layer(model, model_type, tokenizer, attractor, language, templates, layer, layer_list, neuron_list, k, out_dir)
        elif args.algo == 'topkall':
            top_k_all_seq(model, model_type, tokenizer, attractor, language, templates, layer_list, neuron_list, k, out_dir)
    elif (args.algo == 'greedy') and (layer == -1):
        greedy(model, model_type, tokenizer, attractor, templates, k, out_dir)
    elif (args.algo == 'greedy') and (layer != -1):
        greedy_by_layer(model, model_type, tokenizer, attractor, templates, layer, k, out_dir)
    elif (args.algo == 'test'):
        test()
    else:
        random_greedy_by_layer(layer, k, out_dir)