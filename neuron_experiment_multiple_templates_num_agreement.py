
from datetime import datetime
import os
import sys
import random

from utils_num_agreement import convert_results_to_pd
from experiment_num_agreement import Intervention, Model
from transformers import (
    GPT2Tokenizer, TransfoXLTokenizer, XLNetTokenizer, BertTokenizer,
    XLMRobertaTokenizer, CamembertTokenizer, XGLMTokenizer
)
from vocab_utils import get_nouns, get_nouns2, get_verbs, get_verbs2, get_prepositions, \
        get_preposition_nouns, get_adv1s, get_adv2s
from generate_all_sentences import load_nouns, load_verbs, load_nouns2, load_verbs2, load_bigrams, \
        load_adv1, load_adv2, load_prepositions, load_preposition_nouns
import vocab_utils as vocab

'''
Run all the extraction for a model across many templates
'''
LANG_COMPLEMENTIZERS = {
    'fr': 'que',
    'nl': 'die',
    # German inflects the complementizer for case and number
    'de': {'M': 'den', 'N': 'das', 'F': 'die'},
    # Finnish inflects the complementizer for case and number
    'fi': {'P': 'jota', 'E': 'josta', 'I': 'johon',
            'Al': 'jolle', 'Ac': 'jonka'}
}

def get_intervention_types():
    return ['indirect', 'direct']

def construct_templates_fr(language, attractor):
    if "_" in language:
        lang_key = language.split("_")[0]
    else:
        lang_key = language
    
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
        for p in load_prepositions(lang_key):
            for ppns, ppnp in load_preposition_nouns(language):
                ppn = ppns if attractor == 'prep_singular' else ppnp
                if lang_key in ("fr", "nl", "de"):
                    template = ' '.join(['{}', '{}', p, ppn])
                elif lang_key == "fi":
                    # Finnish has postpositions w/ genitive nouns
                    template = ' '.join(['{}', ppn, p])
                templates.append(template)
    elif attractor in ('within_rc_singular', 'within_rc_plural'):
        for ns, np in load_nouns(language):
            noun = ns if attractor.startswith('within_rc_singular') else np
            # template = ' '.join(['The', noun, 'that', 'the', '{}'])
            if lang_key in ("fr", "nl"):
                template = ' '.join([noun, LANG_COMPLEMENTIZERS[lang_key], '{}', '{}'])
            elif lang_key == "fi":
                template = ' '.join([noun, '{}', '{}'])
            elif lang_key == "de":
                template = ' '.join([noun, '{}', '{}', '{}'])
            templates.append(template)
    elif attractor in ['rc_singular', 'rc_plural']:
        for noun2s, noun2p in load_nouns2(language):
            noun2 = noun2s if attractor.startswith('rc_singular') else noun2p
            for verb2s, verb2p in load_verbs2(language):
                if lang_key == "fi":
                    case, verb2s = verb2s.split("_")
                verb2 = verb2s if attractor.startswith('rc_singular') else verb2p
                if lang_key in ("fr", "nl"):
                    template = ' '.join(['{}', '{}', LANG_COMPLEMENTIZERS[lang_key], noun2, verb2])
                elif lang_key == "fi":
                    template = ' '.join(['{}', LANG_COMPLEMENTIZERS['fi'][case], noun2, verb2])
                elif lang_key == "de":  # fill in complementizer later
                    template = ' '.join(['{}', '{}', '{}', noun2, verb2])
                else:
                    raise ValueError("Invalid language.")
                templates.append(template)
    elif attractor == "distractor":
        if language == "fr":
            raise ValueError("Cannot place adverbs before a verb in French.")
        for adv1 in load_adv1(lang_key):
            for adv2 in load_adv2(lang_key):
                templates.append(' '.join(['{}', '{}', adv1, LANG_CONJUNCTIONS[language], adv2]))
    elif attractor == "distractor_1":
        if language == "fr":
            raise ValueError("Cannot place adverbs before a verb in French.")
        for adv1 in load_adv1(lang_key):
            templates.append(' '.join(['{}', '{}', adv1]))
    else:
        templates = ["{} {}"] if lang_key in ("fr", "nl", "de") else ["{}"]
    return templates

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

def construct_interventions_bi(structure, tokenizer, DEVICE, seed, examples, shuffle=False, intervention_method = "natural"):
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    if structure == "semantic":
        structure = "semantic_short"

    word1_list, word2_list = load_bigrams(structure)

    if structure.startswith("bigram"):
        temp = "{}"
    elif structure == "semantic_short":
        temp = "The {}"     # the <adj> <noun>
        word1_list, word2_list = word2_list, word1_list
    elif structure == "semantic_long":
        temp = "The {} is"  # the <noun> is <adj>
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
        except Exception as e:
            pass
    print(f"\t Only used {used_word_count}/{all_word_count} nouns due to tokenizer")
    if examples > 0 and len(interventions) >= examples:     # randomly sample input sentences
        random.seed(seed)
        interventions = {k: v 
                for k, v in random.sample(interventions.items(), examples)}
    return interventions

def construct_interventions_fr(tokenizer, DEVICE, attractor, seed, examples, language, intervention_method = "natural"):
    if "_" in language:
        lang_key = language.split("_")[0]
    else:
        lang_key = language
    
    NOM_TO_ACC = {
        'der': 'den',
        'das': 'das',
        'die': 'die'
    }
    
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    templates = construct_templates_fr(language, attractor)
    for temp in templates:
        if attractor.startswith('within_rc'):
            for noun2s, noun2p in load_nouns2(language):
                temp_mod = temp.capitalize()
                if lang_key in ("fr", "nl"):
                    temp_mod = temp.format(noun2s.split()[0], "{}")
                    noun2s = noun2s.split()[1]
                    noun2p = noun2p.split()[1]
                elif lang_key == "de":
                    complementizer = NOM_TO_ACC[temp.split()[0]]
                    temp_mod = temp.format(complementizer, noun2s.split()[0], "{}")
                    noun2s = noun2s.split()[1]
                    noun2p = noun2p.split()[1]
                if intervention_method == "controlled":
                    noun_list = [noun2s]
                else:
                    noun_list = [noun2s, noun2p]
                for v_singular, v_plural in load_verbs2(language):
                    if language == "fi":
                        case, v_singular = v_singular.split("_")
                        temp_mod = temp.format(LANG_COMPLEMENTIZERS['fi'][case], "{}")
                    all_word_count += 1
                    try:
                        intervention_name = '_'.join([temp, noun2s, v_singular])
                        interventions[intervention_name] = Intervention(
                            tokenizer,
                            temp_mod,
                            noun_list,
                            [v_singular, v_plural],
                            device=DEVICE,
                            method=intervention_method)
                        used_word_count += 1
                    except Exception as e:
                        pass
        else:
            for ns, np in load_nouns(language):
                if lang_key == "de" and attractor.startswith('rc'):
                    complementizer = NOM_TO_ACC[ns.split()[0]]
                    temp_mod = temp.format(ns.capitalize().split()[0], "{}", complementizer)
                elif lang_key in ("fr", "nl", "de"):
                    temp_mod = temp.format(ns.capitalize().split()[0], "{}")
                else:
                    temp_mod = temp
                if lang_key in ("fr", "nl", "de"):
                    ns = ns.split()[1]
                    np = np.split()[1]
                if intervention_method == "controlled":
                    noun_list = [ns]
                else:
                    noun_list = [ns, np]
                for v_singular, v_plural in load_verbs(language):
                    all_word_count += 1
                    try:
                        intervention_name = '_'.join([temp, ns, v_singular])
                        interventions[intervention_name] = Intervention(
                            tokenizer,
                            temp_mod,
                            noun_list,
                            [v_singular, v_plural],
                            device=DEVICE,
                            method=intervention_method)
                        used_word_count += 1
                    except Exception as e:
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
    templates = construct_templates(attractor)
    for temp in templates:
        if attractor.startswith('within_rc'):
            for noun2s, noun2p in get_nouns2():
                if intervention_method == "controlled":
                    noun_list = [noun2s]
                else:
                    noun_list = [noun2s, noun2p]
                for v_singular, v_plural in vocab.get_verbs2():
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

def run_all(model_type="gpt2", attractor=None, intervention_method = "natural", device="cuda", 
            out_dir=".", random_weights=False, seed=5, examples=100, language="en"):
    print("Model:", model_type)
    # Set up all the potential combinations
    intervention_types = get_intervention_types()
    # Initialize Model and Tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    model = Model(device=device, gpt2_version=model_type, 
        random_weights=random_weights)
    tokenizer = (GPT2Tokenizer if model.is_gpt2 else
                 TransfoXLTokenizer if model.is_txl else
                 XLNetTokenizer if model.is_xlnet else
                 BertTokenizer if model.is_bert else
                 XLMRobertaTokenizer if model.is_xlmr else
                 CamembertTokenizer if model.is_camembert else
                 XGLMTokenizer).from_pretrained(model_type)
    # Set up folder if it does not exist
    dt_string = datetime.now().strftime("%Y%m%d")
    folder_name = dt_string+"_neuron_intervention"
    base_path = os.path.join(out_dir, "results", folder_name)
    if random_weights:
        base_path = os.path.join(base_path, "random")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if language != "en":
        interventions = construct_interventions_fr(tokenizer, device, attractor, seed,
                examples, language, intervention_method)
    elif attractor.startswith("bigram") or attractor.startswith("semantic"):
        interventions = construct_interventions_bi(attractor, tokenizer, device, seed, examples,
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
        
        fcomponents = random + [str(attractor), language, itype, model_type, intervention_method]
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
            if value.lower().startswith("f"):
                value = False
            else:
                value = bool(temp[1])
        else:
            value = temp[1]
        default[keyword] = value
    '''
    device = sys.argv[2] # cpu vs cuda
    out_dir = sys.argv[3] # dir to write results
    random_weights = sys.argv[4] == 'random' # true or false
    attractor = sys.argv[5]
    seed = int(sys.argv[6]) # to allow consistent sampling
    examples = int(sys.argv[7]) # number of examples to try, 0 for all 
    if len(sys.argv) > 8:
        language = sys.argv[8]
    else:
        language = "en"
    if len(sys.argv) > 9:
        intervention_method = sys.argv[9]
    else:
        intervention_method = "natural"
    
    run_all(model, attractor, intervention_method, device, out_dir, random_weights, seed, examples, language)
    '''
    run_all(default['model'], default['attractor'], default['intervention_method'],
            default['device'], default['out_dir'], default['random_weights'], default['seed'], default['examples'], default['language'])