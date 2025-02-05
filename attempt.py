from nltk.corpus import wordnet as wn
from collections import defaultdict
import csv
import torch
import torch.nn as nn
import random
random.seed(0)
from sklearn.model_selection import train_test_split
import itertools

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
    def __init__(self, num_letters, hidden_size=128):
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
models = []
for l in range(1, 16):
    input = None
    output = None
    for i in range(len(words_by_length[l])):
        word = words_by_length[l][i]
        word_copy = word
        letters = list(set(list(word)))
        subset = random.sample(letters, random.randint(1, len(letters)))
        for letter in subset:
            word = word.replace(letter, '_')
        next_row_input = torch.zeros(1, l * 27)
        next_row_output = torch.zeros(1, 26)
        for j in range(len(word)):
            if word[j] != "_":
                next_row_input[0, j * 27 + ord(word[j]) - ord('a')] = 1
            else:
                next_row_input[0, j * 27 + 26] = 1
                next_row_output[0, ord(word_copy[j]) - ord('a')] = 1
        if input is None:
            input = next_row_input
            output = next_row_output
        else:
            input = torch.concatenate((input, next_row_input), dim=0)
            output = torch.concatenate((output, next_row_output), dim=0)
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=0)

    # create instance of model
    model = HangmanNN(l)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    losses = []
    print(f'Length {l}')
    print('Training...')
    for epoch in range(epochs):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch} loss is {loss}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_eval = model.forward(X_test)
        loss = criterion(y_eval, y_test)
        print(f'Loss on test set is {loss}')
    
    models.append(model)

# create game playing agent
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def play_hangman(model, word, max_guesses=6):
    n = len(word)
    guesses = 0
    correct_guesses = 0
    guessed_word = ['_'] * n
    guessed_letters = []
    while guesses < max_guesses and correct_guesses < n:
        input = torch.zeros(1, n * 27)
        for i in range(n):
            if guessed_word[i] != '_':
                input[0, i * 27 + ord(guessed_word[i]) - ord('a')] = 1
            else:
                input[0, i * 27 + 26] = 1
        output = model.forward(input)
        output = sigmoid(output)
        temp = [(i, output[0, i].item()) for i in range(26)]
        j = 0
        guess = chr(temp[j][0] + ord('a'))
        while guess in guessed_letters:
            j += 1
            guess = chr(temp[j][0] + ord('a'))
        guessed_letters.append(guess)
        if guess in word:
            for i in range(n):
                if word[i] == guess:
                    guessed_word[i] = guess
                    correct_guesses += 1
        else:
            guesses += 1
    return correct_guesses == n

population = [i for i in range(len(words))]
sample = 100
sampled_population = random.sample(population, sample)
print(sampled_population)
for i in range(sample):
    print(f'Game {i+1}')
    correct = 0
    for j in sampled_population:
        n = len(words[j])
        if play_hangman(models[n-1], words[j]):
            correct += 1
print(f'Accuracy: {correct / sample}')