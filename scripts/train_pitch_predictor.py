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
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        hparams = HyperParams(**config)
    else:
        # Initialize default hparams with dummy values to satisfy assertion
        # These will be overwritten or unused
        dummy_kwargs = {
            'training_files': 'dummy_train.txt',
            'validation_files': 'dummy_val.txt',
            'output_directory': args.output_dir,
            'language': 'english',
            'speakers': ['dummy']
        }
        hparams = HyperParams(verbose=False, **dummy_kwargs)
        
    # Override hparams with args if provided
    if args.data_set_dir:
        # We need to find training files from data_set_dir
        # This logic mimics training.py but simplified
        hparams.training_files = os.path.join(args.output_dir, f'train_{hparams.language}.txt')
        hparams.validation_files = os.path.join(args.output_dir, f'validation_{hparams.language}.txt')
        
        # We need to generate these files if they don't exist, or assume they exist?
        # Better: use DaftExprtDataLoader which expects a text file list.
        # If the user passes a directory, we might need to assume a standard structure or run pre-processing.
        # BUT, for simplicity, let's assume the user points to a directory that HAS been pre-processed 
        # and we can find the train.txt there? 
        # Actually, standard Daft-Exprt workflow creates train.txt in the training output dir.
        # Let's assume the user provides the *dataset directory* (containing features) and we scan it?
        # No, DaftExprtDataLoader takes a text file path.
        # Let's look at how training.py does it. It generates train.txt.
        
        # To make this script standalone and simple, let's allow passing the train.txt directly OR 
        # if data_set_dir is passed, look for standard train.txt location?
        # The user command was: --data_set_dir .../combined_datasets_features_full
        # This directory likely contains the features.
        # We need the train.txt list.
        
        # Let's change the script to accept --training_files directly OR try to find it.
        # But the user wants to run it like: --data_set_dir ...
        # Let's assume standard Daft-Exprt structure: data_set_dir contains features.
        # But where is the split?
        
        # Let's just update hparams with what we have and rely on hparams logic if possible.
        # But hparams needs training_files.
        pass

    # Update batch size and lr
    if args.batch_size:
        hparams.batch_size = args.batch_size
    
    lr = args.lr if args.lr else 1e-4
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # Initialize model
    model = PitchPredictor(n_mel_channels=hparams.n_mel_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Data Loader
    # If training_files is not set in hparams (e.g. no config file), we need it.
    if not hparams.training_files:
        if args.data_set_dir:
             # Try to find train.txt in output_dir (if we ran pre-process there) or just list files?
             # Simplest: Just list all .npy files in data_set_dir recursively?
             # DaftExprtDataLoader expects a file with paths.
             # Let's create a temporary train file from the data_set_dir.
             
             # Find all .npy files (mel specs)
             # Actually, let's just ask the user for the training file list if they don't provide a config.
             # OR, we can assume the user has run pre-processing and we can find the list.
             
             # Wait, the user command provided --data_set_dir pointing to features.
             # Let's assume we can just use all files in there.
             # We can generate a file list on the fly.
             
             train_list_path = os.path.join(args.output_dir, 'train_pitch_predictor.txt')
             with open(train_list_path, 'w') as f:
                 for root, dirs, files in os.walk(args.data_set_dir):
                     for file in files:
                         if file.endswith('.npy') and 'mel_spec' in file:
                             # We need the base path (without _mel_spec.npy)
                             base_path = os.path.join(root, file.replace('-mel_spec.npy', ''))
                             f.write(f"{base_path}\n")
             hparams.training_files = train_list_path
             _logger.info(f"Generated training list at {train_list_path}")
        else:
            raise ValueError("Must provide --config_file or --data_set_dir")

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
    parser.add_argument('--config_file', type=str, default=None, help='Path to config.json')
    parser.add_argument('--data_set_dir', type=str, default=None, help='Path to dataset directory (features)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
