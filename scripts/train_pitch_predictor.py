import argparse
import json
import logging
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ROOT directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.environ['PYTHONPATH'] = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from daft_exprt.data_loader import DaftExprtDataLoader, DaftExprtDataCollate
from daft_exprt.hparams import HyperParams
from daft_exprt.layers.pitch_predictor import PitchPredictor

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

def train(args):
    # Load hparams
    config_file = args.config_file
    with open(config_file, 'r') as f:
        config = json.load(f)
    hparams = HyperParams(**config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # Initialize model
    model = PitchPredictor(n_mel_channels=hparams.n_mel_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Data Loader
    train_dataset = DaftExprtDataLoader(hparams.training_files, hparams)
    collate_fn = DaftExprtDataCollate(hparams)
    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, 
                              num_workers=4, collate_fn=collate_fn, drop_last=True)
    
    _logger.info(f"Starting training on device: {device}")
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            # Parse batch
            # symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths, \
            # frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids, feature_dirs, feature_files = batch
            
            frames_pitch = batch[7].to(device).float() # (B, T)
            mel_specs = batch[8].to(device).float()    # (B, n_mel, T)
            output_lengths = batch[9].to(device).long() # (B, )
            
            # Forward
            pitch_pred = model(mel_specs) # (B, T)
            
            # Masking
            # Create mask from output_lengths
            max_len = pitch_pred.size(1)
            mask = torch.arange(max_len).to(device).expand(len(output_lengths), max_len) < output_lengths.unsqueeze(1)
            
            # Loss
            loss = criterion(pitch_pred * mask, frames_pitch * mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                _logger.info(f"Epoch {epoch} | Iter {i} | Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        _logger.info(f"Epoch {epoch} Complete | Avg Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"pitch_predictor_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            _logger.info(f"Saved checkpoint to {ckpt_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Path to config.json')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
