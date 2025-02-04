from nltk.corpus import wordnet as wn
from collections import defaultdict

# count the distribution of words starting with each letter in this corpus
statistics = defaultdict(int)

words = []
for word in wn.words():
    # filter out words including hyphens, numbers, etc.
    if word.isalpha():
        words.append(word)
        statistics[word[0]] += 1

words.sort()
print(words)
print("Number of words in the corpus: ", len(words))
print("Distribution of words starting with each letter in corpus: ", statistics)


