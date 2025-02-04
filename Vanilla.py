import torch
import torch.nn as nn

# n = len(word)
# n is variable
n = 7
input_dim = 27 * n
# hidden_dim: can experiment with different hidden dimensions
hidden_dim = 32
output_dim = 26

# activation function: can experiment with different activation functions
relu = nn.Relu()

# model architecture
fc1 = nn.Linear(input_dim, hidden_dim)
fc2 = nn.Linear(hidden_dim, output_dim)

softmax = nn.Softmax()

model = nn.Sequential(
    fc1,
    relu,
    fc2,
    softmax
)

def forward(self, x):
    output = self.model(x)
    return output