import torch.nn as nn

input_dim = 10
hidden_size = 20

model_LSTM = nn.LSTM(input_dim, hidden_size, batch_first=True)