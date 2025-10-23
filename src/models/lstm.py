import torch, torch.nn as nn


class LSTMVol(nn.Module):
    def __init__(self, n_feats: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_feats, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus()) # enforce σ ≥ 0
    def forward(self, x): # x: (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)