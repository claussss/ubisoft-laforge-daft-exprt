import torch
from daft_exprt.layers.pitch_predictor import PitchPredictor

def test_shape():
    model = PitchPredictor(n_mel_channels=80, hidden_dim=256, kernel_size=3)
    # Batch=2, Channels=80, Time=100
    x = torch.randn(2, 80, 100)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    if y.shape[1] == x.shape[2]:
        print("SUCCESS: Output length matches input length.")
    else:
        print(f"FAILURE: Output length {y.shape[1]} does not match input length {x.shape[2]}.")

if __name__ == "__main__":
    test_shape()
