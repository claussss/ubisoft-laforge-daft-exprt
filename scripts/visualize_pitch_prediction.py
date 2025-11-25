import argparse
import json
import logging
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
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

def visualize(args):
    # Load hparams (minimal setup)
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        hparams = HyperParams(**config)
    elif args.data_set_dir:
        # Generate training file list from data_set_dir
        # We reuse the logic from train_pitch_predictor.py to get a valid file list
        train_list_path = os.path.join(args.output_dir, 'file_list.txt')
        os.makedirs(args.output_dir, exist_ok=True)
        
        _logger.info(f"Scanning {args.data_set_dir} for files...")
        with open(train_list_path, 'w') as f:
            for root, dirs, files in os.walk(args.data_set_dir):
                for file in files:
                    if file.endswith('.npy') and not file.endswith('mel_spec.npy'): 
                        features_dir = root
                        feature_file = os.path.splitext(file)[0]
                        try:
                            speaker_id = int(os.path.basename(root))
                        except ValueError:
                            try:
                                speaker_id = int(feature_file.split('_')[0])
                            except ValueError:
                                speaker_id = 0
                        f.write(f"{features_dir}|{feature_file}|{speaker_id}\n")
        
        # Initialize hparams
        kwargs = {
            'training_files': train_list_path,
            'validation_files': train_list_path,
            'output_directory': args.output_dir,
            'language': 'english',
            'speakers': ['default']
        }
        hparams = HyperParams(verbose=False, **kwargs)
        
        # Load stats
        if args.stats_file and os.path.isfile(args.stats_file):
            with open(args.stats_file, 'r') as f:
                hparams.stats = json.load(f)
        else:
            # Try to find stats.json
            stats_path = os.path.join(args.data_set_dir, 'stats.json')
            if not os.path.isfile(stats_path):
                 stats_path = os.path.join(os.path.dirname(args.data_set_dir.rstrip('/')), 'stats.json')
            
            if os.path.isfile(stats_path):
                with open(stats_path, 'r') as f:
                    hparams.stats = json.load(f)
            else:
                _logger.warning("No stats file found. Using dummy stats.")
                hparams.stats = {}

        # Patch missing speakers
        speakers_found = set()
        with open(train_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    speakers_found.add(int(parts[2]))
        
        for spk_id in speakers_found:
            spk_key = f'spk {spk_id}'
            if spk_key not in hparams.stats:
                hparams.stats[spk_key] = {
                    'energy': {'mean': 0.0, 'std': 1.0},
                    'pitch': {'mean': 0.0, 'std': 1.0}
                }
    else:
        raise ValueError("Must provide --data_set_dir")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # Initialize model
    model = PitchPredictor(n_mel_channels=hparams.n_mel_channels).to(device)
    
    # Load checkpoint
    if args.checkpoint:
        _logger.info(f"Loading checkpoint from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    else:
        _logger.error("Must provide --checkpoint")
        sys.exit(1)
        
    model.eval()
    
    # Data Loader
    dataset = DaftExprtDataLoader(hparams.training_files, hparams)
    collate_fn = DaftExprtDataCollate(hparams)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    _logger.info(f"Generating {args.num_examples} plots...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_examples:
                break
                
            # Parse batch
            # frames_pitch: (B, T)
            # mel_specs: (B, n_mel, T)
            # output_lengths: (B, )
            frames_pitch = batch[7].to(device).float()
            mel_specs = batch[8].to(device).float()
            output_lengths = batch[9].to(device).long()
            
            # Predict
            pitch_pred = model(mel_specs) # (B, T)
            
            # Get data for plotting (take first item in batch since batch_size=1)
            length = output_lengths[0].item()
            gt_pitch = frames_pitch[0, :length].cpu().numpy()
            pred_pitch = pitch_pred[0, :length].cpu().numpy()
            
            # Unvoiced masking for GT (usually 0)
            # We can mask predictions where GT is 0 to see voiced parts clearly
            # Or just plot everything
            
            plt.figure(figsize=(10, 4))
            plt.plot(gt_pitch, label='Ground Truth', alpha=0.7)
            plt.plot(pred_pitch, label='Prediction', alpha=0.7, linestyle='--')
            
            plt.title(f"Pitch Prediction Example {i+1}")
            plt.xlabel("Frames")
            plt.ylabel("Pitch (log Hz)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(args.output_dir, f"pitch_example_{i+1}.png")
            plt.savefig(save_path)
            plt.close()
            _logger.info(f"Saved {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_examples', type=int, default=10)
    parser.add_argument('--stats_file', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    visualize(args)
