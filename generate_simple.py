import os
import csv

PATH="vocab/wordlists/"

def inflect_det(gender, number):
    if number == "plural":
        return "les"
    else:
        if gender == "M":
            return "le"
        elif gender == "F":
            return "la"
        else:
            return "l\'"

nouns = []
with open(PATH + "noun_fr.txt", "r") as nouns:
    for line in nouns:
        nouns.append(line.split())  # format: (singular, plural, class)

with open(PATH + "verb_fr.txt", "r") as verbs:
    for line in verbs:
        verbs.append(line.split())  # format: (singular, plural)

with open("vocab/simple_fr.txt", "w") as simple_out:
    writer = csv.writer(simple_out, delimiter="\t")
    _id = 1
    for noun in nouns:
        for verb in verbs:
            det_s = inflect_det(noun[2], "singular")
            det_p = inflect_det("", "plural")
            # singular sentence
            sent_correct = f"{det_s.capitalize()} {noun[0]} {verb[0]}"
            writer.writerow([sent_correct, "singular", "correct", f"id{_id}"])
            sent_wrong = f"{det_s.capitalize()} {noun[0]} {verb[1]}"
            writer.writerow([sent_wrong, "singular", "wrong", f"id{_id}"])
            _id += 1
            # plural sentence
            sent_correct = f"{det_p.capitalize()} {noun[1]} {verb[1]}"
            writer.writerow([sent_correct, "plural", "correct", f"id{_id}"])
            sent_wrong = f"{det_p.capitalize()} {noun[1]} {verb[0]}"
            writer.writerow([sent_wrong, "plural", "wrong", f"id{_id}"])
            _id += 1
