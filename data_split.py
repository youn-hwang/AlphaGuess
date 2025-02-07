from nltk.corpus import wordnet as wn
import pandas as pd
import random
random.seed(0)

words = [word for word in wn.words() if word.isalpha() and len(word) <= 15]
words.sort()
raw_words = random.sample(words, int(0.80 * len(words)))
test_words = [word for word in words if word not in raw_words]
train_words = random.sample(raw_words, int(0.80 * len(raw_words)))
valid_words = [word for word in raw_words if word not in train_words]

df_words = pd.DataFrame(words, columns=['words'])
df_words.to_csv('words.csv', index=False)

df_train_words = pd.DataFrame(train_words, columns=['train words'])
df_train_words.to_csv('train_words.csv', index=False)

df_valid_words = pd.DataFrame(valid_words, columns=['valid words'])
df_valid_words.to_csv('valid_words.csv', index=False)

df_test_words = pd.DataFrame(test_words, columns=['test words'])
df_test_words.to_csv('test_words.csv', index=False)
print('done')