import torch, torch.nn as nn


class TransVol(nn.Module):
    def __init__(self, n_feats: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128):
        super().__init__()
        self.proj = nn.Linear(n_feats, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
    def forward(self, x):
        z = self.proj(x)
        z = self.encoder(z)
        return self.head(z[:, -1, :]).squeeze(-1)