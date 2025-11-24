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
    elif args.data_set_dir:
        # Generate training file list from data_set_dir
        train_list_path = os.path.join(args.output_dir, 'train_pitch_predictor.txt')
        _logger.info(f"Scanning {args.data_set_dir} for training files...")
        
        with open(train_list_path, 'w') as f:
            for root, dirs, files in os.walk(args.data_set_dir):
                for file in files:
                    if file.endswith('.npy') and not file.endswith('mel_spec.npy'): 
                        # Assuming standard format: SpeakerID/SpeakerID_FileID.npy
                        # e.g. .../0011/0011_000001.npy
                        
                        # We need to write: features_dir|feature_file|speaker_id
                        features_dir = root
                        feature_file = os.path.splitext(file)[0]
                        
                        # Try to extract speaker ID from directory name or filename
                        # Directory name is usually the speaker ID (e.g. 0011)
                        try:
                            speaker_id = int(os.path.basename(root))
                        except ValueError:
                            # Fallback: try first part of filename
                            try:
                                speaker_id = int(feature_file.split('_')[0])
                            except ValueError:
                                speaker_id = 0 # Default if unknown
                        
                        f.write(f"{features_dir}|{feature_file}|{speaker_id}\n")
        
        _logger.info(f"Generated training list at {train_list_path}")
        
        # Initialize hparams with real values
        kwargs = {
            'training_files': train_list_path,
            'validation_files': train_list_path, # Use same for validation as placeholder
            'output_directory': args.output_dir,
            'language': 'english',
            'speakers': ['default'] # Placeholder speakers list
        }
        hparams = HyperParams(verbose=False, **kwargs)
        
        # Load stats.json if available in data_set_dir or parent
        if args.stats_file:
            stats_path = args.stats_file
        else:
            stats_path = os.path.join(args.data_set_dir, 'stats.json')
            if not os.path.isfile(stats_path):
                # Try parent directory
                stats_path = os.path.join(os.path.dirname(args.data_set_dir.rstrip('/')), 'stats.json')
            
        if os.path.isfile(stats_path):
            _logger.info(f"Loading stats from {stats_path}")
            with open(stats_path, 'r') as f:
                hparams.stats = json.load(f)
                
            # Patch stats for missing speakers to prevent KeyError in DataLoader
            # PitchPredictor doesn't use normalized symbols/energy, so dummy stats are fine.
            # We need to scan the training list to find all speakers.
            speakers_found = set()
            with open(train_list_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 3:
                        speakers_found.add(int(parts[2]))
            
            for spk_id in speakers_found:
                spk_key = f'spk {spk_id}'
                if spk_key not in hparams.stats:
                    _logger.warning(f"Speaker {spk_key} missing from stats. Using dummy stats.")
                    hparams.stats[spk_key] = {
                        'energy': {'mean': 0.0, 'std': 1.0},
                        'pitch': {'mean': 0.0, 'std': 1.0}
                    }
        else:
            _logger.warning(f"Could not find stats.json in {args.data_set_dir} or parent. Normalization might fail.")
    else:
        raise ValueError("Must provide --config_file or --data_set_dir")

    # Override hparams with args if provided
    if args.batch_size:
        hparams.batch_size = args.batch_size
    
    lr = args.lr if args.lr else 1e-4
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # Initialize model
    model = PitchPredictor(n_mel_channels=hparams.n_mel_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Load checkpoint if provided
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            _logger.info(f"Loading checkpoint from {args.checkpoint}")
            state_dict = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state_dict)
        else:
            _logger.error(f"Checkpoint file not found: {args.checkpoint}")
            sys.exit(1)
    
    _logger.info(f"Training files path: {hparams.training_files}")
    if not os.path.isfile(hparams.training_files):
        _logger.error(f"Training file does not exist: {hparams.training_files}")

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
    parser.add_argument('--stats_file', type=str, default=None, help='Path to stats.json')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
