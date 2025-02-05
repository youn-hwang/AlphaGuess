from nltk.corpus import wordnet as wn
from collections import defaultdict
import csv
import torch
import torch.nn as nn
import random
random.seed(0)
from sklearn.model_selection import train_test_split

words = []
for word in wn.words():
    # filter out words including hyphens, numbers, etc.
    if word.isalpha() and len(word) <= 15:
        words.append(word)
words.sort()

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

# create neural network
class HangmanNN(nn.Module):
    def __init__(self, num_letters, hidden_size=32):
        super(HangmanNN, self).__init__()
        alphabet = 26
        self.fc1 = nn.Linear(num_letters * (alphabet + 1), hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, alphabet)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# create inputs and outputs for the neural network
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
                input[i, j * 27 + 27] = 1
                output[i, ord(word[j]) - ord('a')] = 1
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=0)

    # create instance of model
    model = HangmanNN(l)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    losses = []
    for epoch in range(epochs):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss)

        if epoch % 10 == 0:
            print(str(l))
            print(f'Epoch {epoch} loss is {loss}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_eval = model.forward(X_test)
        loss = criterion(y_eval, y_test)
        print(f'Loss on test set is {loss}')