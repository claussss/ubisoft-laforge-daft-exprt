import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class ProsodyAdversary(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(ProsodyAdversary, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Predicts Pitch and Energy (2 values)
        )

    def forward(self, x, alpha=1.0):
        """
        x: (B, T, input_dim) - Phoneme embeddings
        alpha: Gradient reversal scaling factor
        """
        x = GradientReversalLayer.apply(x, alpha)
        return self.net(x)
