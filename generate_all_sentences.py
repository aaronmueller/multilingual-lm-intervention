import os
import csv
import argparse

VOCAB_PATH = "vocab/"
WORDS_PATH = "vocab/wordlists/"

templates = {
    "simple": "{} {}",
    "within_rc": "{} que {} {}",
    "rc": "{} que {} {} {}",
    "nounpp": "{} {} {} {}",
    "adv_conjunction": "{} {} et {} {}"
}

def load_bigrams():
    word1_list = []
    word2_list = []
    with open(os.path.join(VOCAB_PATH, "bigrams.txt"), 'r') as bigrams:
        for line in bigrams:
            word1, word2 = line.strip().split()
            word1_list.append(word1)
            word2_list.append(word2)
    return (word1_list, word2_list)

def load_nouns(language):
    nouns_list = []
    with open(os.path.join(WORDS_PATH, f"noun_{language}.txt"), 'r') as nouns:
        for line in nouns:
            nouns_list.append(line.strip().split("\t"))
    return nouns_list

def load_nouns2(language):
    nouns2_list = []
    with open(os.path.join(WORDS_PATH, f"noun2_{language}.txt"), 'r') as nouns2:
        for line in nouns2:
            nouns2_list.append(line.strip().split("\t"))
    return nouns2_list

def load_verbs(language):
    verb_list = []
    with open(os.path.join(WORDS_PATH, f"verb_{language}.txt"), 'r') as verbs:
        for line in verbs:
            verb_list.append(line.strip().split())
    return verb_list

def load_verbs2(language):
    verb2_list = []
    with open(os.path.join(WORDS_PATH, f"verb2_{language}.txt"), 'r') as verbs2:
        for line in verbs2:
            verb2_list.append(line.strip().split())
    return verb2_list

def load_prepositions(language):
    prep_list = []
    with open(os.path.join(WORDS_PATH, f"prep_{language}.txt"), 'r') as preps:
        for line in preps:
            prep_list.append(line.strip())
    return prep_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", type=str, required=True, help="The language "
                        "for which we generate sample sentences.")
    args = parser.parse_args()
    for structure in templates.keys():
        nouns = load_nouns(args.language)
        nouns2 = load_nouns2(args.language)
        verbs = load_verbs(args.language)
        verbs2 = load_verbs(args.language)
        preps = load_prepositions(args.language)

        if structure == "simple":
            _id = 1
            out_file = open(os.path.join(VOCAB_PATH, f"simple_{args.language}.txt"), 'w')
            writer = csv.writer(out_file, delimiter="\t")
            for noun in nouns:
                for verb in verbs:
                    # singular example
                    right_sent = templates[structure].format(noun[0], verb[0])
                    wrong_sent = templates[structure].format(noun[0], verb[1])
                    writer.writerow([right_sent, "singular", "correct", f"id{_id}"])
                    writer.writerow([wrong_sent, "singular", "wrong", f"id{_id}"])
                    _id += 1
                    # plural example
                    right_sent = templates[structure].format(noun[1], verb[1])
                    wrong_sent = templates[structure].format(noun[1], verb[0])
                    writer.writerow([right_sent, "plural", "correct", f"id{_id}"])
                    writer.writerow([wrong_sent, "plural", "wrong", f"id{_id}"])
                    _id += 1
            out_file.close()
            break