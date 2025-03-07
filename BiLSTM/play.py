import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from tqdm import tqdm  # For progress bars

# Set up device and max sequence length (must match training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 15

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

# Load the trained model (trained with MAX_LENGTH = 15)
model = BiLSTMModel(input_dim=27, lstm_units=256, num_layers=3, dropout_rate=0.1,
                    seq_length=MAX_LENGTH, output_dim=26).to(device)
model.load_state_dict(torch.load('best_model_single_2.pth', map_location=device))
model.eval()

# Build a list of models for each possible word length (for evaluation purposes).
# Here we use the same model for all lengths; adjust if you have separate models.
models = [model for _ in range(MAX_LENGTH)]

def play_hangman(model, word, max_guesses=6):
    """
    Plays one game of hangman using the provided model.
    Returns the number of wrong guesses and the final guessed word.
    """
    n = len(word)
    guessed_word = ['_'] * n
    guessed_letters = set()
    wrong_guesses = 0
    correct_guesses = 0

    while '_' in guessed_word and wrong_guesses < max_guesses:
        # Create input matrix for current state.
        inp = np.zeros((n, 27))
        for i in range(n):
            if guessed_word[i] != '_':
                inp[i, ord(guessed_word[i]) - ord('a')] = 1
            else:
                inp[i, 26] = 1
        # Pad input to MAX_LENGTH if necessary.
        if n < MAX_LENGTH:
            pad_width = ((0, MAX_LENGTH - n), (0, 0))
            inp = np.pad(inp, pad_width, mode='constant', constant_values=0)
        inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)
        hidden = model.init_hidden(1)
        logits, _ = model(inp, hidden)  # Unpack output tuple.
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
        sorted_indices = np.argsort(probs)[::-1]
        candidate = None
        for idx in sorted_indices:
            letter = chr(idx + ord('a'))
            if letter not in guessed_letters:
                candidate = letter
                break
        if candidate is None:
            candidate = chr(sorted_indices[0] + ord('a'))
        guessed_letters.add(candidate)
        
        if candidate in word:
            for i, ch in enumerate(word):
                if ch == candidate:
                    guessed_word[i] = candidate
                    correct_guesses += 1
        else:
            wrong_guesses += 1
    return wrong_guesses, ''.join(guessed_word)

# Read test words.
df_test = pd.read_csv("../test_words.csv", keep_default_na=False)
test_words = df_test['test words'].tolist()

# Evaluation on 1000 randomly sampled test words.
sampled_population = random.sample(range(len(test_words)), 1000)
correct_sample = 0
total_wrong_guesses_sample = 0

for idx in tqdm(sampled_population, desc="Evaluating 1000 Sample Games", total=1000):
    word = test_words[idx]
    n = len(word)
    wrong_guesses, final_guess = play_hangman(models[n-1], word)
    if final_guess == word:
        correct_sample += 1
    total_wrong_guesses_sample += wrong_guesses

print(f'\n1000 Sample Evaluation:')
print(f'Accuracy (games fully guessed correctly): {correct_sample / 1000:.4f}')
print(f'Average number of wrong guesses: {total_wrong_guesses_sample / 1000:.4f}')


# Evaluation on all test words.
correct_all = 0
total_wrong_guesses_all = 0
for word in tqdm(test_words, desc="Evaluating All Games"):
    n = len(word)
    wrong_guesses, final_guess = play_hangman(models[n-1], word)
    if final_guess == word:
        correct_all += 1
    total_wrong_guesses_all += wrong_guesses

print(f'\nAll Words Evaluation:')
print(f'Accuracy (games with <= 6 wrong guesses): {correct_all / len(test_words):.4f}')
print(f'Average number of wrong guesses: {total_wrong_guesses_all / len(test_words):.4f}')
