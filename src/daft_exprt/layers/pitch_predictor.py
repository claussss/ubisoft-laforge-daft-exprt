"""Frozen pitch predictor used for pitch consistency loss.

Architecture mirrors the one from simplified_codebase2_code that was used
to train the checkpoint (weight_norm Conv1d, no transpose -- input is
(B, n_mel_channels, T) directly).
"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class _ConvNorm1D(nn.Module):
    """Conv1d with weight_norm and auto same-padding.

    Expects input shape (B, C_in, T), outputs (B, C_out, T).
    Matches the ConvNorm1D used in simplified_codebase2_code/layers so that
    the pre-trained checkpoint state dict loads correctly.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(_ConvNorm1D, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias))

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        return self.conv(signal)


class PitchPredictor(nn.Module):
    """Predict frame-level pitch from a mel-spectrogram.

    Input:  mel_specs  (B, n_mel_channels, T)
    Output: pitch_pred (B, T)
    """

    def __init__(self, n_mel_channels=80, hidden_dim=256, kernel_size=3, dropout=0.1):
        super(PitchPredictor, self).__init__()

        self.conv_layers = nn.Sequential(
            _ConvNorm1D(n_mel_channels, hidden_dim, kernel_size=kernel_size, w_init_gain='relu'),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            _ConvNorm1D(hidden_dim, hidden_dim, kernel_size=kernel_size, w_init_gain='relu'),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            _ConvNorm1D(hidden_dim, hidden_dim, kernel_size=kernel_size, w_init_gain='relu'),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            _ConvNorm1D(hidden_dim, 1, kernel_size=kernel_size, w_init_gain='linear'),
        )

    def forward(self, mel_specs):
        """
        Args:
            mel_specs: (B, n_mel_channels, T)
        Returns:
            pitch_pred: (B, T)
        """
        return self.conv_layers(mel_specs).squeeze(1)
