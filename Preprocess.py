from nltk.corpus import wordnet as wn
from collections import defaultdict

# count the distribution of words 
# starting with each letter in this corpus
# and the distribution of word lengths in this corpus
starting_letter_distribution = defaultdict(int)
length_distribution = defaultdict(int)

words = []
for word in wn.words():
    # filter out words including hyphens, numbers, etc.
    if word.isalpha() and len(word) <= 15:
        words.append(word)
        starting_letter_distribution[word[0]] += 1
        length_distribution[len(word)] += 1

words.sort()
starting_letter_distribution = dict(sorted(starting_letter_distribution.items()))
length_distribution = dict(sorted(length_distribution.items()))

# print(words)
print("Number of words in the corpus: ", len(words))
print("Distribution of words starting with each letter in corpus: ", starting_letter_distribution)
print("Distribution of word lengths in corpus: ", length_distribution)

import csv

# save words to a csv file
url = 'words'
with open(url + '.csv', 'w') as file:
    writer = csv.writer(file)
    for item in words:
        writer.writerow([item])

# words by length
words_by_length = defaultdict(list)
for word in words:
    words_by_length[len(word)].append(word)

for i in range(1, 16):
    with open(url + str(i) + '.csv', 'w') as file:
        writer = csv.writer(file)
        for item in words_by_length[i]:
            writer.writerow([item])

# create inputs and outputs
import torch
import random
random.seed(0)
from sklearn.model_selection import train_test_split

for l in range(1, 16):
    input = torch.zeros(len(words_by_length[l]), l * 27)
    output = torch.zeros(len(words_by_length[l]), 26)
    for i in range(len(words_by_length[l])):
        word = words_by_length[l][i]
        index = random.randint(0, len(word)-1)
        letter = word[index]
        word.replace(letter, '_')
        for j in range(len(word)):
            if word[j] != "_":
                input[i, j * 27 + ord(word[j]) - ord('a')] = 1
            else:
                output[i, ord(word[j]) - ord('a')] = 1
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=0)
