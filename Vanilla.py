import torch
import torch.nn as nn
import torch.nn.functional as F

class HangmanNN(nn.Module):
    def __init__(self, num_letters, hidden_size=32):
        super(HangmanNN, self).__init__()
        alphabet = 26
        self.fc1 = nn.Linear(num_letters * (alphabet + 1), hidden_size)
        self.relu = nn.ReLu()
        self.fc2 = nn.Linear(hidden_size, alphabet)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)

# create instance of model
model = HangmanNN(5)