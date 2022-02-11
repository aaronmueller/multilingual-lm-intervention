
from datetime import datetime
import os
import sys
import random

from utils_num_agreement import convert_results_to_pd
from experiment_num_agreement import Intervention, Model
from transformers import (
    GPT2Tokenizer, TransfoXLTokenizer, XLNetTokenizer, BertTokenizer,
    XLMRobertaTokenizer, CamembertTokenizer
)
from vocab_utils import get_nouns, get_nouns2, get_verbs, get_verbs2, get_prepositions, \
        get_preposition_nouns, get_adv1s, get_adv2s
from generate_all_sentences import load_nouns, load_verbs, load_nouns2, load_verbs2, load_bigrams, \
        load_adv1, load_adv2, load_prepositions, load_preposition_nouns
import vocab_utils as vocab

'''
Run all the extraction for a model across many templates
'''

def get_intervention_types(intervetion_method = "natural"):
    if intervetion_method == "natural":
        return ['indirect', 'direct']
    else:
        return ['indirect']

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

def construct_interventions_bi(tokenizer, DEVICE, seed, examples, shuffle=False, intervention_method = "natural"):
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
                device=DEVICE,
                method = intervention_method)
            used_word_count += 1
        except AssertionError as e:
            pass
    print(f"\t Only used {used_word_count}/{all_word_count} nouns due to tokenizer")
    if examples > 0 and len(interventions) >= examples:     # randomly sample input sentences
        random.seed(seed)
        interventions = {k: v 
                for k, v in random.sample(interventions.items(), examples)}
    return interventions

def construct_interventions_fr(tokenizer, DEVICE, attractor, seed, examples, language, intervention_method = "natural"):
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
                            if intervention_method == "controlled":
                                interventions[intervention_name] = Intervention(
                                    tokenizer,
                                    temp,
                                    [ns],#, np],
                                    [v_singular, v_plural],
                                    device=DEVICE,
                                    method = intervention_method)
                            else:
                                interventions[intervention_name] = Intervention(
                                    tokenizer,
                                    temp,
                                    [ns, np],
                                    [v_singular, v_plural],
                                    device=DEVICE,
                                    method = intervention_method)
                            used_word_count += 1
                        else:
                            if intervention_method == "controlled":
                                interventions[intervention_name] = Intervention(
                                    tokenizer,
                                    temp.format(ns.capitalize().split()[0], "{}"),
                                    [ns.split()[1]],#, np.split()[1]],
                                    [v_singular, v_plural],
                                    device=DEVICE,
                                    method = intervention_method)
                            else:
                                interventions[intervention_name] = Intervention(
                                    tokenizer,
                                    temp.format(ns.capitalize().split()[0], "{}"),
                                    [ns.split()[1], np.split()[1]],
                                    [v_singular, v_plural],
                                    device=DEVICE,
                                    method = intervention_method)
                            used_word_count += 1
                    except AssertionError as e:
                        pass
    print(f"\t Only used {used_word_count}/{all_word_count} nouns due to tokenizer")
    if examples > 0 and len(interventions) >= examples:     # randomly sample input sentences
        random.seed(seed)
        interventions = {k: v 
                for k, v in random.sample(interventions.items(), examples)}
    return interventions

def construct_interventions(tokenizer, DEVICE, attractor, seed, examples, intervention_method = "natural"):
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    templates = construct_templates()
    for temp in templates:
        if attractor.startswith('within_rc'):
            for noun2s, noun2p in get_nouns2():
                for v_singular, v_plural in vocab.get_verbs():
                    all_word_count += 1
                    try:
                        intervention_name = '_'.join([temp, noun2s, v_singular])
                        if intervention_method == "controlled":
                            interventions[intervention_name] = Intervention(
                                tokenizer,
                                temp,
                                [noun2s],#, noun2p],
                                [v_singular, v_plural],
                                device=DEVICE,
                                method = intervention_method)
                        else:
                            interventions[intervention_name] = Intervention(
                                tokenizer,
                                temp,
                                [noun2s, noun2p],
                                [v_singular, v_plural],
                                device=DEVICE,
                                method = intervention_method)
                        used_word_count += 1
                    except AssertionError as e:
                        pass
        else:
            for ns, np in vocab.get_nouns():
                for v_singular, v_plural in vocab.get_verbs():
                    all_word_count += 1
                    try: 
                        intervention_name = '_'.join([temp, ns, v_singular])
                        if intervention_method == "controlled":
                            interventions[intervention_name] = Intervention(
                                tokenizer,
                                temp,
                                [ns],#, noun2p],
                                [v_singular, v_plural],
                                device=DEVICE,
                                method = intervention_method)
                        else:
                            interventions[intervention_name] = Intervention(
                                tokenizer,
                                temp,
                                [ns, np],
                                [v_singular, v_plural],
                                device=DEVICE,
                                method = intervention_method)
                        used_word_count += 1
                    except AssertionError as e:
                        pass
    print(f"\t Only used {used_word_count}/{all_word_count} nouns due to tokenizer")
    if examples > 0 and len(interventions) >= examples:     # randomly sample input sentences
        random.seed(seed)
        interventions = {k: v 
                for k, v in random.sample(interventions.items(), examples)}
    return interventions

def run_all(model_type="gpt2", attractor=None, intervention_method = "natural", device="cuda", 
            out_dir=".", random_weights=False, seed=5, examples=100, language="en"):
    print("Model:", model_type)
    # Set up all the potential combinations
    intervention_types = get_intervention_types(intervention_method)
    # Initialize Model and Tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    model = Model(device=device, gpt2_version=model_type, 
        random_weights=random_weights)
    tokenizer = (GPT2Tokenizer if model.is_gpt2 else
                 TransfoXLTokenizer if model.is_txl else
                 XLNetTokenizer if model.is_xlnet else
                 BertTokenizer if model.is_bert else
                 XLMRobertaTokenizer if model.is_xlmr else
                 CamembertTokenizer).from_pretrained(model_type)
    # Set up folder if it does not exist
    dt_string = datetime.now().strftime("%Y%m%d")
    folder_name = dt_string+"_neuron_intervention"
    base_path = os.path.join(out_dir, "results", folder_name)
    if random_weights:
        base_path = os.path.join(base_path, "random")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    print(language)
    if language != "en":
        interventions = construct_interventions_fr(tokenizer, device, attractor, seed,
                examples, language, intervention_method)
    elif attractor.startswith("bigram"):
        interventions = construct_interventions_bi(tokenizer, device, seed, examples,
                shuffle=attractor.endswith("shuffle"), intervention_method = intervention_method)
    else:
        interventions = construct_interventions(tokenizer, device, attractor, seed,
                examples, intervention_method)
    # Consider all the intervention types
    for itype in intervention_types:
        print("\t Running with intervention: {}".format(
            itype))
        # Run actual exp
        intervention_results = model.neuron_intervention_experiment(
            interventions, itype, alpha=1.0)

        df = convert_results_to_pd(interventions, intervention_results)
        # Generate file name
        random = ['random'] if random_weights else []
        if '/' in model_type:
            model_type = model_type.split('/')[1]
        if language != "en":
            fcomponents = random + [str(attractor), language, itype, model_type]
        else:
            fcomponents = random + [str(attractor), itype, model_type]
        fname = "_".join(fcomponents)
        # Finally, save each exp separately
        df.to_csv(os.path.join(base_path, fname+".csv"))


if __name__ == "__main__":
    if not (len(sys.argv) >= 4):
        print("USAGE: python ", sys.argv[0], 
"<model> <attractor> <intervention_method> (<device> <out_dir> <random_weights> <seed> <examples> <language>)")
    model = sys.argv[1] # distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl
    attractor = sys.argv[2] # singular, plural or none
    intervention_method = sys.argv[3] # natural, controlled
    default = {'model':model, 'attractor': attractor, 'intervention_method': intervention_method,
                'device':'cuda', 'out_dir':'.', 'random_weights': False, 
                'seed':3, 'examples':200, 'language':'en'}
    
    for arg in sys.argv[4:]:
        temp = arg.split(':')
        keyword = temp[0]
        value = temp[1]
        if keyword in ['seed', 'examples']:
            value = int(temp[1])
        elif keyword in ['random_weights']:
            value = bool(temp[1])
        else:
            value = temp[1]
        default[keyword] = value
    '''
    device = sys.argv[2] # cpu vs cuda
    out_dir = sys.argv[3] # dir to write results
    random_weights = sys.argv[4] == 'random' # true or false
    seed = int(sys.argv[6]) # to allow consistent sampling
    examples = int(sys.argv[7]) # number of examples to try, 0 for all 
    if len(sys.argv) > 8:
        language = sys.argv[8]
    else:
        language = "en"
    '''
    run_all(default['model'], default['attractor'], default['intervention_method'],
            default['device'], default['out_dir'], default['random_weights'], default['seed'], default['examples'], default['language'])