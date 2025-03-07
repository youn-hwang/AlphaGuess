import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm 

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

url = "../"

df_words = pd.read_csv(url + 'words.csv', keep_default_na=False)
df_train_words = pd.read_csv(url + 'train_words.csv', keep_default_na=False)
df_valid_words = pd.read_csv(url + 'valid_words.csv', keep_default_na=False)
words = df_words['words'].tolist()
train_words = df_train_words['train words'].tolist()
valid_words = df_valid_words['valid words'].tolist()
print('Imported words...')

train_words_by_length = defaultdict(list)
valid_words_by_length = defaultdict(list)
for word in words:
    if word in train_words:
        train_words_by_length[len(word)].append(word)
    if word in valid_words:
        valid_words_by_length[len(word)].append(word)

def masking(word, mask):
    return "".join(['_' if letter in mask else letter for letter in word])

torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, seq_length):
        super(AttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, 1))
        self.b = nn.Parameter(torch.zeros(seq_length, 1))
    def forward(self, x):
        et = torch.tanh(torch.matmul(x, self.W) + self.b)
        et = et.squeeze(-1)
        at = F.softmax(et, dim=1)
        at = at.unsqueeze(-1)
        output = x * at
        return output.sum(dim=1)

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, lstm_units, num_layers, dropout_rate, seq_length, output_dim):
        super(BiLSTMModel, self).__init__()
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(input_size=input_dim, hidden_size=lstm_units, 
                                        batch_first=True, bidirectional=True))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        for _ in range(1, num_layers):
            self.lstm_layers.append(nn.LSTM(input_size=2*lstm_units, hidden_size=lstm_units, 
                                            batch_first=True, bidirectional=True))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        self.attention = AttentionLayer(hidden_dim=2*lstm_units, seq_length=seq_length)
        self.fc = nn.Linear(2*lstm_units, output_dim)
    def forward(self, x, hidden=None):
        new_hidden = []
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            if hidden is not None:
                h0, c0 = hidden[i]
                h0 = h0.to(x.device)
                c0 = c0.to(x.device)
                x, (hn, cn) = lstm(x, (h0, c0))
                new_hidden.append((hn, cn))
            else:
                x, (hn, cn) = lstm(x)
            x = dropout(x)
        x = self.attention(x)
        x = self.fc(x)
        if hidden is not None:
            return x, new_hidden
        return x
    def init_hidden(self, batch_size):
        hidden_states = []
        for lstm in self.lstm_layers:
            num_directions = 2
            h0 = torch.zeros(num_directions, batch_size, lstm.hidden_size).to(device)
            c0 = torch.zeros(num_directions, batch_size, lstm.hidden_size).to(device)
            hidden_states.append((h0, c0))
        return hidden_states

max_length = max(len(word) for word in train_words)
X_train = []
y_train = []
X_val = []
y_val = []

# Prepare training data.
for word in train_words:
    l = len(word)
    all_letters = list(set(word))
    size = random.randint(1, len(all_letters))
    mask = random.sample(all_letters, size)
    masked_word = masking(word, mask)
    inp = np.zeros((l, 27))
    target = np.zeros((26,))
    for j in range(l):
        if masked_word[j] != "_":
            inp[j, ord(word[j]) - ord('a')] = 1
        else:
            inp[j, 26] = 1
            target[ord(word[j]) - ord('a')] = 1
    if l < max_length:
        pad_width = ((0, max_length - l), (0, 0))
        inp = np.pad(inp, pad_width, mode='constant', constant_values=0)
    X_train.append(inp)
    y_train.append(target)

for word in valid_words:
    l = len(word)
    all_letters = list(set(word))
    size = random.randint(1, len(all_letters))
    mask = random.sample(all_letters, size)
    masked_word = masking(word, mask)
    inp = np.zeros((l, 27))
    target = np.zeros((26,))
    for j in range(l):
        if masked_word[j] != "_":
            inp[j, ord(word[j]) - ord('a')] = 1
        else:
            inp[j, 26] = 1
            target[ord(word[j]) - ord('a')] = 1
    if l < max_length:
        pad_width = ((0, max_length - l), (0, 0))
        inp = np.pad(inp, pad_width, mode='constant', constant_values=0)
    X_val.append(inp)
    y_val.append(target)

X_train = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
y_train = torch.tensor(np.array(y_train), dtype=torch.float32).to(device)
X_val = torch.tensor(np.array(X_val), dtype=torch.float32).to(device)
y_val = torch.tensor(np.array(y_val), dtype=torch.float32).to(device)

# Increase dropout rate to 0.3.
model = BiLSTMModel(input_dim=27, lstm_units=256, num_layers=3, dropout_rate=0.3, 
                    seq_length=max_length, output_dim=26).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

batch_size = 32
num_epochs = 50
patience = 3
best_val_loss = float('inf')
epochs_without_improvement = 0

torch.autograd.set_detect_anomaly(True)

train_losses = []
val_losses = []

# Learning rate scheduler reduces LR on plateau.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    permutation = torch.randperm(X_train.shape[0])
    
    pbar = tqdm(range(0, X_train.shape[0], batch_size), desc=f"Epoch {epoch+1}", leave=False)
    
    for i in pbar:
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        x_batch = X_train[indices]
        y_batch = y_train[indices]
        hidden = model.init_hidden(x_batch.shape[0])
        
        outputs, _ = model(x_batch, hidden)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix(loss=loss.item())
    
    avg_epoch_loss = epoch_loss / num_batches
    train_losses.append(avg_epoch_loss)
    
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden(X_val.shape[0])
        val_outputs, _ = model(X_val, hidden)
        val_loss = criterion(val_outputs, y_val).item()
        val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}: Train Loss {avg_epoch_loss:.4f}, Val Loss {val_loss:.4f}")
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model_single_2.pth')
        print("Validation loss improved. Saving model checkpoint.")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("BiLSTM_loss_plot_2.png")
plt.close()
