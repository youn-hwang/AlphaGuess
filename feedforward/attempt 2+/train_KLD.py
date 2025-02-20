from collections import defaultdict
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
random.seed(0)

# Import words
url = "feedforward/attempt 2+/"
df_words = pd.read_csv(url + 'words.csv', keep_default_na=False)
words = df_words['words'].tolist()

df_train_words = pd.read_csv(url + 'train_words.csv', keep_default_na=False)
train_words = df_train_words['train words'].tolist()

df_valid_words = pd.read_csv(url + 'valid_words.csv', keep_default_na=False)
valid_words = df_valid_words['valid words'].tolist()
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
        out = self.relu(self.fc1(x))
        out = F.log_softmax(self.fc2(out), dim=1)
        return out

def masking(word, mask):
    return "".join(['_' if letter in mask else letter for letter in word])

# Create inputs and outputs for the neural network
models = []
train_losses = []
validation_losses = []
for l in range(1, 16):
    X_train = []
    y_train = []
    embedding_train = {}
    for word in train_words_by_length[l]:
        word_original = word
        all_letters = list(set(list(word)))
        for size in range(1, len(all_letters) + 1):
            for mask in itertools.combinations(all_letters, size):
                masked_word = masking(word, mask)
                if masked_word in embedding_train:
                    output = embedding_train[masked_word]
                else:
                    output = np.zeros((26, ))
                for j in range(len(word)):
                    if masked_word[j] == "_":
                        output[ord(word[j]) - ord('a')] += 1
                embedding_train[masked_word] = output
    for masked_word in embedding_train:
        next_row_input = np.zeros((l * 27, ))
        for j in range(len(word)):
            if masked_word[j] != "_":
                next_row_input[j * 27 + ord(word[j]) - ord('a')] = 1
            else:
                next_row_input[j * 27 + 26] = 1
        X_train.append(next_row_input)
        output = embedding_train[masked_word]
        y_train.append(F.softmax(torch.tensor(output, dtype=torch.float32), dim=0))                

    X_val = []
    y_val = []
    embedding_val = {}
    for word in valid_words_by_length[l]:
        word_original = word
        all_letters = list(set(list(word)))
        for size in range(1, len(all_letters) + 1):
            for mask in itertools.combinations(all_letters, size):
                masked_word = masking(word, mask)
                if masked_word in embedding_val:
                    output = embedding_val[masked_word]
                else:
                    output = np.zeros((26, ))
                for j in range(len(masked_word)):
                    if masked_word[j] == "_":
                        output[ord(word[j]) - ord('a')] += 1
                embedding_val[masked_word] = output
    for masked_word in embedding_val:
        next_row_input = np.zeros((l * 27, ))
        for j in range(len(word)):
            if masked_word[j] != "_":
                next_row_input[j * 27 + ord(word[j]) - ord('a')] = 1
            else:
                next_row_input[j * 27 + 26] = 1
        X_val.append(next_row_input)
        output = embedding_val[masked_word]
        y_val.append(F.softmax(torch.tensor(output, dtype=torch.float32), dim=0)) 

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    model = HangmanNN(l, 128)
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    num_epochs = 10
    num_batches = 10
    n = X_train.size()[0]
    batch_size = n // num_batches
    train_loss = 0
    for epoch in range(num_epochs):
        if batch_size == 0:
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            print(f'Length {l}, Epoch {epoch}, Loss: {loss.item()}')
        else:
            for i in range(0, n, batch_size):
                X_train_batch  = X_train[i: min(i+batch_size, n)]
                y_train_batch = y_train[i: min(i+batch_size, n)]
                optimizer.zero_grad()
                y_pred = model(X_train_batch)
                loss = criterion(y_pred, y_train_batch)
                loss.backward()
                optimizer.step()
                print(f'Length {l}, Epoch {epoch}, Batch, {i//batch_size}, Loss: {loss.item()}')
        train_loss = loss.item()
    train_losses.append(train_loss)

    model.eval()
    validation_loss = 0
    with torch.no_grad():
        y_pred = model(X_val)
        val_loss = criterion(y_pred, y_val)
        print(f'Length {l}, Validation Loss: {val_loss.item()}')
        validation_loss = val_loss.item()
    validation_losses.append(validation_loss)
    
    models.append(model)

# Plot losses
plt.plot(train_losses, label='train')
plt.plot(validation_losses, label='validation')
plt.legend()
plt.show()