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
    """Replaces any letter found in 'mask' with '_'."""
    return "".join(['_' if letter in mask else letter for letter in word])


torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        x = token_emb + pos_emb  # (batch_size, seq_length, embed_dim)
        
        x = x.transpose(0, 1)
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        encoded = encoded.transpose(0, 1)  # (batch_size, seq_length, embed_dim)
        

        attn_scores = self.attn_pool(encoded)  
        attn_weights = torch.softmax(attn_scores, dim=1)  
        pooled = (encoded * attn_weights).sum(dim=1)  # 
        
        logits = self.fc(pooled)  
        return logits

def encode_and_mask(word, max_length):
   
    all_letters = list(set(word))
    # Randomly select some letters to mask (at least one)
    size = random.randint(1, len(all_letters))
    mask_letters = random.sample(all_letters, size)
    
    masked_word = masking(word, mask_letters)
    
    input_seq = []
    target = np.zeros((26,), dtype=np.float32)
    
    for j, orig_letter in enumerate(word):
        if masked_word[j] != '_':
            input_seq.append(ord(orig_letter) - ord('a'))
        else:
            input_seq.append(26)
            target[ord(orig_letter) - ord('a')] = 1
    

    if len(input_seq) < max_length:
        input_seq.extend([26] * (max_length - len(input_seq)))
    return input_seq, target


max_length = max(len(word) for word in train_words)

X_train, y_train = [], []
X_val, y_val = [], []


for word in train_words:
    inp, target = encode_and_mask(word, max_length)
    X_train.append(inp)
    y_train.append(target)

# Prepare validation data
for word in valid_words:
    inp, target = encode_and_mask(word, max_length)
    X_val.append(inp)
    y_val.append(target)

# Convert lists to tensors
X_train = torch.tensor(np.array(X_train), dtype=torch.long).to(device)
y_train = torch.tensor(np.array(y_train), dtype=torch.float32).to(device)
X_val = torch.tensor(np.array(X_val), dtype=torch.long).to(device)
y_val = torch.tensor(np.array(y_val), dtype=torch.float32).to(device)

#Model
model = MaskedCharacterTransformer(vocab_size=27, embed_dim=256, num_heads=16, num_layers=4,
                                     max_seq_length=max_length, ff_dim=256, dropout=0.1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training
batch_size = 32
num_epochs = 50
patience = 3
best_val_loss = float('inf')
epochs_without_improvement = 0

torch.autograd.set_detect_anomaly(True)

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    permutation = torch.randperm(X_train.shape[0])
    
    for i in range(0, X_train.shape[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        x_batch = X_train[indices]
        y_batch = y_train[indices]
        
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_epoch_loss = epoch_loss / num_batches
    train_losses.append(avg_epoch_loss)
    
    # Validation evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}: Train Loss {avg_epoch_loss:.4f}, Val Loss {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model_transformer.pth')
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
plt.savefig("MaskedCharacterTransformer_loss_plot.png")
plt.close()
