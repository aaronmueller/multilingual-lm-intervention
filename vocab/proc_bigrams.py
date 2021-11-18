import csv

with open("bigrams.csv", "r") as bigrams, open("bigrams.txt", "w") as out_bigrams:
    reader = csv.reader(bigrams)
    for row in reader:
        word1, word2, category, _id = row
        out_bigrams.write(word1 + " " + word2 + "\n")
