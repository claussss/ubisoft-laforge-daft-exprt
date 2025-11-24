import torch
import torch.nn as nn
from daft_exprt.layers import ConvNorm1D

class PitchPredictor(nn.Module):
    def __init__(self, n_mel_channels=80, hidden_dim=256, kernel_size=3, dropout=0.1):
        super(PitchPredictor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            ConvNorm1D(n_mel_channels, hidden_dim, kernel_size=kernel_size, w_init_gain='relu'),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            ConvNorm1D(hidden_dim, hidden_dim, kernel_size=kernel_size, w_init_gain='relu'),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            ConvNorm1D(hidden_dim, hidden_dim, kernel_size=kernel_size, w_init_gain='relu'),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            ConvNorm1D(hidden_dim, 1, kernel_size=kernel_size, w_init_gain='linear')
        )

    def forward(self, mel_specs):
        """
        mel_specs: (B, n_mel_channels, T)
        return: (B, T)
        """
        # (B, 1, T)
        pitch_pred = self.conv_layers(mel_specs)
        # (B, T)
        return pitch_pred.squeeze(1)
