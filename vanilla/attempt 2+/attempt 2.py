from collections import defaultdict
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
random.seed(0)

# Import words
url = "vanilla/attempt 2/"
df_words = pd.read_csv(url + 'words.csv', keep_default_na=False)
words = df_words['words'].tolist()

df_train_words = pd.read_csv(url + 'train_words.csv', keep_default_na=False)
train_words = df_train_words['train words'].tolist()

df_valid_words = pd.read_csv(url + 'valid_words.csv', keep_default_na=False)
valid_words = df_valid_words['valid words'].tolist()

df_test_words = pd.read_csv(url + 'test_words.csv', keep_default_na=False)
test_words = df_test_words['test words'].tolist()
print('imported words...')

# Organize words by length
train_words_by_length = defaultdict(list)
valid_words_by_length = defaultdict(list)
for word in words:
    if word in train_words:
        train_words_by_length[len(word)].append(word)
    if word in valid_words:
        valid_words_by_length[len(word)].append(word)

# Neural network definition
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

# Create inputs and outputs for the neural network
models = []
for l in range(1, 16):
    X_train = []
    y_train = []
    for word in train_words_by_length[l]:
        word_copy = word
        my_list = list(set(list(word)))
        for k in range(2):
            size = random.randint(1, len(my_list))
            subset = random.sample(my_list, size)
            for letter in subset:
                word = word.replace(letter, '_')
            next_row_input = np.zeros((l * 27,))
            next_row_output = np.zeros((26,))
            for j in range(len(word)):
                if word[j] != "_":
                    next_row_input[j * 27 + ord(word[j]) - ord('a')] = 1
                else:
                    next_row_input[j * 27 + 26] = 1
                    next_row_output[ord(word_copy[j]) - ord('a')] = 1
            X_train.append(next_row_input)
            y_train.append(next_row_output)
    X_val = []
    y_val = []
    for word in valid_words_by_length[l]:
        word_copy = word
        my_list = list(set(list(word)))
        for k in range(1):
            size = random.randint(1, len(my_list))
            subset = random.sample(my_list, size)
            for letter in subset:
                word = word.replace(letter, '_')
            next_row_input = np.zeros((l * 27,))
            next_row_output = np.zeros((26,))
            for j in range(len(word)):
                if word[j] != "_":
                    next_row_input[j * 27 + ord(word[j]) - ord('a')] = 1
                else:
                    next_row_input[j * 27 + 26] = 1
                    next_row_output[ord(word_copy[j]) - ord('a')] = 1
            X_val.append(next_row_input)
            y_val.append(next_row_output)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    model = HangmanNN(l, 1000)
    criterion = nn.BCEWithLogitsLoss()
    # RMSProp instead of Adam gives similar results
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    model.train()
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    
        print(f'Length {l}, Epoch {epoch}, Loss: {loss.item()}')
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val)
        val_loss = criterion(y_pred, y_val)
        print(f'Length {l}, Validation Loss: {val_loss.item()}')
    
    models.append(model)

# Game playing agent
def play_hangman(model, word, max_guesses=6):
    n = len(word)
    guesses = 0
    correct_guesses = 0
    guessed_word = ['_'] * n
    guessed_letters = []
    while correct_guesses < n:
        input = torch.zeros(1, n * 27)
        for i in range(n):
            if guessed_word[i] != '_':
                input[0, i * 27 + ord(guessed_word[i]) - ord('a')] = 1
            else:
                input[0, i * 27 + 26] = 1
        output = model(input)
        temp = [(i, output[0, i].item()) for i in range(26)]
        temp.sort(key=lambda x: x[1], reverse=True)
        guess = chr(temp[0][0] + ord('a'))
        for i in range(1, 26):
            if guess not in guessed_letters:
                break
            guess = chr(temp[i][0] + ord('a'))
        guessed_letters.append(guess)
        if guess in word:
            for i in range(n):
                if word[i] == guess:
                    guessed_word[i] = guess
                    correct_guesses += 1
        else:
            guesses += 1
    return guesses

# Evaluate the model
sampled_population = random.sample(range(len(valid_words)), 1000)
correct = 0
total_guesses = 0
for i in range(1000):
    word = test_words[sampled_population[i]]
    n = len(word)
    guesses = play_hangman(models[n-1], word)
    if guesses <= 6:
        correct += 1
    total_guesses += guesses
    # print(f'Game {i+1}, Word: {word}, Guesses: {guesses}')
print(f'Accuracy: {correct / 1000}')
print(f'Average number of guesses: {total_guesses / 1000}')