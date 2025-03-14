import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 15  
class MaskedCharacterTransformer(nn.Module):
    def __init__(self, vocab_size=27, embed_dim=256, num_heads=16, num_layers=4,
                                     max_seq_length=15, ff_dim=256, dropout=0.1):
        super(MaskedCharacterTransformer, self).__init__()
        self.max_seq_length = max_seq_length
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_seq_length, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=ff_dim, 
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = nn.Linear(embed_dim, 1)
        self.fc = nn.Linear(embed_dim, 26)
        
    def forward(self, x, src_key_padding_mask=None):
        batch_size, seq_length = x.size()
        token_emb = self.token_embedding(x)  
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        pos_emb = self.positional_embedding(positions)
        x = token_emb + pos_emb  
        
        x = x.transpose(0, 1)
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        encoded = encoded.transpose(0, 1)  
        
        attn_scores = self.attn_pool(encoded)           
        attn_weights = torch.softmax(attn_scores, dim=1)  
        pooled = (encoded * attn_weights).sum(dim=1)   
        
        logits = self.fc(pooled)  
        return logits

def encode_word(word, max_length):
    token_ids = []
    for ch in word:
        if ch == '_':
            token_ids.append(26)
        else:
            token_ids.append(ord(ch) - ord('a'))
    # Pad if necessary
    if len(token_ids) < max_length:
        token_ids.extend([26] * (max_length - len(token_ids)))
    return torch.tensor(token_ids, dtype=torch.long)

#Load
model = MaskedCharacterTransformer(vocab_size=27, embed_dim=256, num_heads=16, num_layers=4,
                                     max_seq_length=15, ff_dim=256, dropout=0.1).to(device)
model.load_state_dict(torch.load('best_model_transformer.pth', map_location=device))
model.eval()

def play_hangman(model, word, max_guesses=6):
    n = len(word)
    guessed_word = ['_'] * n
    guessed_letters = set()
    wrong_guesses = 0
    
    while '_' in guessed_word and wrong_guesses < max_guesses:
        current_state = ''.join(guessed_word)
        input_tensor = encode_word(current_state, MAX_LENGTH).unsqueeze(0).to(device)
        logits = model(input_tensor)  # shape: (1, 26)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        sorted_indices = np.argsort(probs)[::-1]  # descending order of probabilities
        
        # highest probability
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
        else:
            wrong_guesses += 1
    
    return wrong_guesses, ''.join(guessed_word)

#Test Set Evaluation
df_test = pd.read_csv("../test_words.csv", keep_default_na=False)
test_words = df_test['test words'].tolist()

sampled_population = random.sample(range(len(test_words)), 1000)
correct_sample = 0
total_wrong_guesses_sample = 0

for idx in tqdm(sampled_population, desc="Evaluating 1000 Sample Games", total=1000):
    word = test_words[idx]
    wrong_guesses, final_guess = play_hangman(model, word)
    if final_guess == word:
        correct_sample += 1
    total_wrong_guesses_sample += wrong_guesses

print(f'\n1000 Sample Evaluation:')
print(f'Accuracy (games fully guessed correctly): {correct_sample / 1000:.4f}')
print(f'Average number of wrong guesses: {total_wrong_guesses_sample / 1000:.4f}')

correct_all = 0
total_wrong_guesses_all = 0
for word in tqdm(test_words, desc="Evaluating All Games"):
    wrong_guesses, final_guess = play_hangman(model, word)
    if final_guess == word:
        correct_all += 1
    total_wrong_guesses_all += wrong_guesses

print(f'\nAll Words Evaluation:')
print(f'Accuracy (games with <= 6 wrong guesses): {correct_all / len(test_words):.4f}')
print(f'Average number of wrong guesses: {total_wrong_guesses_all / len(test_words):.4f}')
