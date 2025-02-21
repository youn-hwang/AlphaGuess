import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import defaultdict

random.seed(0)

# Define file path
url = "rnn/"

# Import words
df_words = pd.read_csv(url + 'words.csv', keep_default_na=False)
df_train_words = pd.read_csv(url + 'train_words.csv', keep_default_na=False)
df_valid_words = pd.read_csv(url + 'valid_words.csv', keep_default_na=False)

words = df_words['words'].tolist()
train_words = df_train_words['train words'].tolist()
valid_words = df_valid_words['valid words'].tolist()
print('Imported words...')

# Organize words by length
train_words_by_length = defaultdict(list)
valid_words_by_length = defaultdict(list)
for word in words:
    if word in train_words:
        train_words_by_length[len(word)].append(word)
    if word in valid_words:
        valid_words_by_length[len(word)].append(word)

# Define RNN Model
class HangmanRNN(nn.Module):
    def __init__(self, hidden_size=128):
        super(HangmanRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(27, hidden_size, batch_first=True)  
        self.fc = nn.Linear(hidden_size, 26)  # No sigmoid here

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  # Taking last time-step output
        return out, hidden  # Logits (no sigmoid)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)  # Ensure tensor is on correct device

# Masking Function
def masking(word, mask):
    return "".join(['_' if letter in mask else letter for letter in word])

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models, train_losses, validation_losses = [], [], []
for l in range(1, 16):
    X_train, y_train, X_val, y_val = [], [], [], []

    # Generate Training Data
    for word in train_words_by_length[l]:
        all_letters = list(set(list(word)))
        for _ in range(2):
            size = random.randint(1, len(all_letters))
            mask = random.sample(all_letters, size)
            masked_word = masking(word, mask)
            
            next_row_input = np.zeros((l, 27))
            next_row_output = np.zeros((26,))
            for j in range(len(word)):
                if masked_word[j] != "_":
                    next_row_input[j, ord(word[j]) - ord('a')] = 1
                else:
                    next_row_input[j, 26] = 1
                    next_row_output[ord(word[j]) - ord('a')] = 1
            X_train.append(next_row_input)
            y_train.append(next_row_output)

    # Generate Validation Data
    for word in valid_words_by_length[l]:
        all_letters = list(set(list(word)))
        for _ in range(1):
            size = random.randint(1, len(all_letters))
            mask = random.sample(all_letters, size)
            masked_word = masking(word, mask)

            next_row_input = np.zeros((l, 27))
            next_row_output = np.zeros((26,))
            for j in range(len(word)):
                if masked_word[j] != "_":
                    next_row_input[j, ord(word[j]) - ord('a')] = 1
                else:
                    next_row_input[j, 26] = 1
                    next_row_output[ord(word[j]) - ord('a')] = 1
            X_val.append(next_row_input)
            y_val.append(next_row_output)

    # Convert to Tensors
    X_train, y_train = torch.tensor(np.array(X_train), dtype=torch.float32), torch.tensor(np.array(y_train), dtype=torch.float32)
    X_val, y_val = torch.tensor(np.array(X_val), dtype=torch.float32), torch.tensor(np.array(y_val), dtype=torch.float32)

    # Move to GPU
    X_train, y_train, X_val, y_val = X_train.to(device), y_train.to(device), X_val.to(device), y_val.to(device)

    # Initialize Model
    model = HangmanRNN().to(device)
    criterion = nn.BCEWithLogitsLoss()  # No need for sigmoid
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training
    batch_train = X_train.shape[0]
    epochs = 100
    for epoch in range(epochs):
        hidden = model.init_hidden(batch_train)  # Fix: Removed extra .detach()

        model.train()
        optimizer.zero_grad()
        output, hidden = model(X_train, hidden)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        print(f'Length {l}, Epoch {epoch}, Loss: {loss.item()}')
        train_loss = loss.item()
    
    train_losses.append(train_loss)

    # Validation
    batch_val = X_val.shape[0]
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden(batch_val)  # Fix: Removed extra .detach()
        y_pred, _ = model(X_val, hidden)
        val_loss = criterion(y_pred, y_val)
        print(f'Length {l}, Validation Loss: {val_loss.item()}')
        validation_losses.append(val_loss.item())

    models.append(model)

# Plot Losses
plt.plot(range(1, 16), train_losses, label='Train')
plt.plot(range(1, 16), validation_losses, label='Validation')
plt.legend()
plt.xlabel("Length of word")
plt.ylabel("Loss value")
plt.show()