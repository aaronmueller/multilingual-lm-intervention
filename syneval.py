import torch
import random
import sys
# import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import random
from functools import partial
from tqdm import tqdm
# from tqdm import tqdm_notebook
import math
import statistics
import vocab_utils as vocab

from utils_num_agreement import batch, convert_results_to_pd
from experiment_num_agreement import Model
from neuron_experiment_multiple_templates_num_agreement import construct_interventions_bi, construct_interventions_fr, construct_templates
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    TransfoXLTokenizer,
    XLNetTokenizer,
    BertForMaskedLM, BertTokenizer,
    XLMRobertaForMaskedLM, XLMRobertaTokenizer,
    CamembertForMaskedLM, CamembertTokenizer
)
from transformers_modified.modeling_transfo_xl import TransfoXLLMHeadModel
from transformers_modified.modeling_xlnet import XLNetLMHeadModel
from attention_intervention_model import (
    AttentionOverride, TXLAttentionOverride, XLNetAttentionOverride, BertAttentionOverride
)
from vocab_utils import get_nouns, get_nouns2, get_verbs, get_verbs2, get_prepositions, \
        get_preposition_nouns, get_adv1s, get_adv2s
from generate_all_sentences import load_nouns, load_verbs, load_nouns2, load_verbs2, load_bigrams, \
        load_adv1, load_adv2, load_prepositions, load_preposition_nouns
import vocab_utils as vocab

PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

class Intervention():
    '''
    Wrapper for all the possible interventions
    '''
    def __init__(self,
                 tokenizer,
                 base_string: str,
                 substitutes: list,
                 candidates: list,
                 device='cpu'):
        super()
        self.device = device
        self.enc = tokenizer

        if isinstance(tokenizer, XLNetTokenizer):
            base_string = PADDING_TEXT + ' ' + base_string
        # All the initial strings
        # First item should be neutral, others tainted
        self.base_strings = [base_string.format(s)
                             for s in substitutes]
        # Tokenized bases
        #self.base_strings_tok = [self.enc.encode(s)
        #                         for s in self.base_strings]
        # print(self.base_strings_tok)
        #self.base_strings_tok = torch.LongTensor(self.base_strings_tok)\
        #                             .to(device)
        self.base_string_tok = self.enc.encode(self.base_strings[0], add_special_tokens=False,
                                               add_space_before_punct_symbol=True)
        self.alt_string_tok = self.enc.encode(self.base_strings[1], add_special_tokens=False,
                                              add_space_before_punct_symbol=True)
        #self.base_strings_tok = [
        #    self.enc.encode(s, add_special_tokens=False, 
        #    add_space_before_punct_symbol=True)
        #    for s in self.base_strings
        #]

        self.base_string_tok = torch.LongTensor(self.base_string_tok).to(device)
        self.alt_string_tok = torch.LongTensor(self.alt_string_tok).to(device)
        # Where to intervene
        #self.position = base_string.split().index('{}')
        if isinstance(tokenizer, XLNetTokenizer):
            diff = len(base_string.split()) - base_string.split().index('{}')
            self.position = len(self.base_strings_tok[0]) - diff
            assert len(self.base_strings_tok[0]) == len(self.base_strings_tok[1])
        else:
            self.position = base_string.split().index('{}')

        self.candidates = []
        for c in candidates:
            # 'a ' added to input so that tokenizer understand that first word
            # follows a space.
            # tokens = self.enc.tokenize('. ' + c)[1:] 
            tokens = self.enc.tokenize('a ' + c,
                add_space_before_punct_symbol=True)[1:]
            assert(len(tokens) == 1)
            self.candidates.append(tokens)

        self.substitutes = []
        for s in substitutes:
            # 'a ' added to input so that tokenizer understand that first word
            # follows a space.
            tokens = self.enc.tokenize('a ' + s,
                add_space_before_punct_symbol=True)[1:]
            self.substitutes.append(tokens)
        assert(len(self.substitutes[0]) > 1 and len(self.substitutes[0]) == len(self.substitutes[1]))
        # self.position += len(self.substitutes[0]) - 1

        self.candidates_tok = [self.enc.convert_tokens_to_ids(tokens)
                               for tokens in self.candidates]

def construct_templates_fr(language):
    LANG_COMPLEMENTIZERS = {
        'fr': 'que',
        'nl': 'die',
        # German inflects the complementizer for case and number
        'de': {'M': 'den', 'N': 'das', 'F': 'die'},
        # Finnish inflects the complementizer for case and number
        'fi': {'P': 'jota', 'E': 'josta', 'I': 'johon',
               'Al': 'jolle', 'Ac': 'jonka'}
    }

    LANG_CONJUNCTIONS = {
        'fr': 'et',
        'nl': 'en',
        'de': 'und',
        'fi': 'ja'
    }

    templates = []
    if attractor in ['prep_singular', 'prep_plural']:
        for p in load_prepositions(language):
            for ppns, ppnp in load_preposition_nouns(language):
                ppn = ppns if attractor == 'prep_singular' else ppnp
                if language in ("fr", "nl"):
                    template = ' '.join(['{}', '{}', p, ppn])
                elif language == "fi":
                    # Finnish has postpositions w/ genitive nouns
                    template = ' '.join(['{}', ppn, p])
                templates.append(template)
    elif attractor in ['rc_singular', 'rc_plural']:
        for noun2s, noun2p in load_nouns2(language):
            noun2 = noun2s if attractor.startswith('rc_singular') else noun2p
            for verb2s, verb2p in load_verbs2(language):
                if language == "fi":
                    case, verb2s = verb2s.split("_")
                verb2 = verb2s if attractor.startswith('rc_singular') else verb2p
                if language in ("fr", "nl"):
                    template = ' '.join(['{}', '{}', LANG_COMPLEMENTIZERS[language], noun2, verb2])
                elif language == "fi":
                    template = ' '.join(['{}', LANG_COMPLEMENTIZERS['fi'][case], noun2, verb2])
                else:
                    raise ValueError("Invalid language.")
                templates.append(template)
    elif attractor == "distractor":
        if language == "fr":
            raise ValueError("Cannot place adverbs before a verb in French.")
        for adv1 in load_adv1(language):
            for adv2 in load_adv2(language):
                templates.append(' '.join(['{}', '{}', adv1, LANG_CONJUNCTIONS[language], adv2]))
    elif attractor == "distractor_1":
        if language == "fr":
            raise ValueError("Cannot place adverbs before a verb in French.")
        for adv1 in load_adv1(language):
            templates.append(' '.join(['{}', '{}', adv1]))
    else:
        templates = ["{} {}"] if language in ("fr", "nl") else ["{}"]
    return templates

def construct_templates():
    # specify format of inputs. fill in with terminals later
    templates = []
    if attractor in  ['singular', 'plural']:
        for p in vocab.get_prepositions():
            for ppns, ppnp in vocab.get_preposition_nouns():
                ppn = ppns if attractor == 'singular' else ppnp
                template = ' '.join(['The', '{}', p, 'the', ppn])
                templates.append(template)
    elif attractor in ('rc_singular', 'rc_plural', 'rc_singular_no_that', 'rc_plural_no_that'):
        for noun2s, noun2p in vocab.get_nouns2():
            noun2 = noun2s if attractor.startswith('rc_singular') else noun2p
            for verb2s, verb2p in vocab.get_verbs2():
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
        for  adv1 in vocab.get_adv1s():
            for adv2 in vocab.get_adv2s():
                templates.append(' '.join(['The', '{}', adv1, 'and', adv2]))
    elif attractor == 'distractor_1':
        for adv1 in vocab.get_adv1s():
            templates.append(' '.join(['The', '{}', adv1]))

    else:   # defaults to simple agreement
        templates = ['The {}']
    return templates

def construct_interventions_bi(tokenizer, DEVICE, seed, shuffle=False):
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    temp = "{}"
    word1_list, word2_list = load_bigrams()
    if shuffle:
        random.shuffle(word1_list)
        random.shuffle(word2_list)
    for (idx, (word1, word2)) in enumerate(zip(word1_list, word2_list)):
        if idx + 1 >= len(word1_list):
            break
        all_word_count += 1
        try:
            intervention_name = '_'.join([temp, word1, word2])
            interventions[intervention_name] = Intervention(
                tokenizer,
                temp,
                [word1, word1_list[idx+1]],
                [word2, word2_list[idx+1]],
                device=DEVICE)
            used_word_count += 1
        except AssertionError as e:
            pass
    '''
    print(f"\t Only used {used_word_count}/{all_word_count} nouns due to tokenizer")
    if examples > 0 and len(interventions) >= examples:     # randomly sample input sentences
        random.seed(seed)
        interventions = {k: v 
                for k, v in random.sample(interventions.items(), examples)}
    '''
    interventions = [v for k, v in interventions.items()]
    return interventions

def construct_interventions_fr(tokenizer, DEVICE, attractor, seed, language):
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    templates = construct_templates_fr(language)
    for temp in templates:
        if attractor.startswith('within_rc'):
            pass
        else:
            for ns, np in load_nouns(language):
                for v_singular, v_plural in load_verbs(language):
                    all_word_count += 1
                    try:
                        intervention_name = '_'.join([temp, ns, v_singular])
                        if language == "fi":
                            interventions[intervention_name] = Intervention(
                                tokenizer,
                                temp,
                                [ns, np],
                                [v_singular, v_plural],
                                device=DEVICE)
                            used_word_count += 1
                        else:
                            interventions[intervention_name] = Intervention(
                                tokenizer,
                                temp.format(ns.capitalize().split()[0], "{}"),
                                [ns.split()[1], np.split()[1]],
                                [v_singular, v_plural],
                                device=DEVICE)
                            used_word_count += 1
                    except Exception as e:
                        pass
    '''
    print(f"\t Only used {used_word_count}/{all_word_count} nouns due to tokenizer")
    if examples > 0 and len(interventions) >= examples:     # randomly sample input sentences
        random.seed(seed)
        interventions = {k: v 
                for k, v in random.sample(interventions.items(), examples)}
    '''
    interventions = [v for k, v in interventions.items()]
    return interventions

def construct_interventions(tokenizer, DEVICE, attractor, seed):
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    templates = construct_templates()
    for temp in templates:
        if attractor.startswith('within_rc'):
            for noun2s, noun2p in vocab.get_nouns2():
                for v_singular, v_plural in vocab.get_verbs():
                    all_word_count += 1
                    intervention_name = '_'.join([temp, noun2s, v_singular])
                    interventions[intervention_name] = Intervention(
                        tokenizer,
                        temp,
                        [noun2s, noun2p],
                        [v_singular, v_plural],
                        device=DEVICE)
                    used_word_count += 1
        else:
            for ns, np in vocab.get_nouns():
                for v_singular, v_plural in vocab.get_verbs():
                    try:
                        all_word_count += 1
                        intervention_name = '_'.join([temp, ns, v_singular])
                        interventions[intervention_name] = Intervention(
                            tokenizer,
                            temp,
                            [ns, np],
                            [v_singular, v_plural],
                            device=DEVICE)
                        used_word_count += 1
                    except AssertionError as e:
                        pass
    #print(f"\t Only used {used_word_count}/{all_word_count} nouns due to tokenizer")
    #if examples > 0 and len(interventions) >= examples:     # randomly sample input sentences
    #    random.seed(seed)
    interventions = [v for k, v in interventions.items()]
    #    interventions = {k: v 
    #            for k, v in random.sample(interventions.items(), examples)}
    return interventions


def syneval(model_type, device, random_weights, attractor, seed):
    print("Model:", model_type)
    # Initialize Model and Tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    model = Model(device=device, gpt2_version=model_type, 
        random_weights=random_weights)
    tokenizer = (GPT2Tokenizer if model.is_gpt2 else
                 TransfoXLTokenizer if model.is_txl else
                 XLNetTokenizer if model.is_xlnet else
                 BertTokenizer if model.is_bert else
                 XLMRobertaTokenizer).from_pretrained(model_type)
    # Set up folder if it does not exist
    if language != "en":
        interventions = construct_interventions_fr(tokenizer, device, attractor, seed, language)
    elif attractor.startswith("bigram"):
        interventions = construct_interventions_bi(tokenizer, device, seed, shuffle=attractor.endswith("shuffle"))
    else:
        interventions = construct_interventions(tokenizer, device, attractor, seed)
    #interventions = construct_interventions(tokenizer, device, attractor, seed)
    n_correct_singular = 0
    n_correct_plural = 0
    n_multitoken_increase = 0
    total = 0
    for i in tqdm(range(len(interventions))):
        intervention = interventions[i]
        if len(intervention.candidates[0]) != len(intervention.candidates[1]):
        # if len(intervention.candidates[0]) != 1 or len(intervention.candidates[1]) != 1:
            continue
        # candidate1_base_prob, candidate2_base_prob = model.get_probabilities_for_examples_multitoken(
        candidate1_base_prob, candidate2_base_prob = model.get_probabilities_for_examples(
                intervention.base_string_tok.unsqueeze(0),
                intervention.candidates_tok)[0]
        # candidate1_alt_prob, candidate2_alt_prob = model.get_probabilities_for_examples_multitoken(
        candidate1_alt_prob, candidate2_alt_prob = model.get_probabilities_for_examples(
            intervention.alt_string_tok.unsqueeze(0),
            intervention.candidates_tok)[0]
        total += 1
        if candidate1_base_prob > candidate2_base_prob:
            n_correct_singular += 1
        if candidate2_alt_prob > candidate1_alt_prob:
            n_correct_plural += 1
        if candidate2_alt_prob > candidate2_base_prob:
            n_multitoken_increase += 1
    print("% Correct (singular): {}".format(n_correct_singular / total))
    print("% Correct (plural): {}".format(n_correct_plural / total))
    print("% Probability increase: {}".format(n_multitoken_increase / total))
    print("Total examples: {}".format(total))
        

if __name__ == "__main__":
    if not (len(sys.argv) >= 6):
        print("USAGE: python ", sys.argv[0], 
            "<model> <device> <random_weights> <attractor> <seed>")
    model = sys.argv[1] # distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl
    device = sys.argv[2] # cpu vs cuda
    random_weights = sys.argv[3] == 'random' # true or false
    attractor = sys.argv[4] # singular, plural or none
    seed = int(sys.argv[5]) # to allow consistent sampling
    if len(sys.argv) > 6:
        language = sys.argv[6]
    else:
        language = "en"
    
    syneval(model, device, random_weights, attractor, seed)