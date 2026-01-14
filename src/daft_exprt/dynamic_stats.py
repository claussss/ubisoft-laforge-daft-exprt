import torch
import numpy as np
import random
import logging
from collections import defaultdict
import os

_logger = logging.getLogger(__name__)

class DynamicSpeakerStatsManager:
    """
    Manages dynamic speaker statistics and embeddings computation during training.
    
    It maintains a subset of "support set" files for each speaker. 
    Periodically refreshes this subset.
    Computes mean/std of pitch/energy and average speaker embedding from the subset.
    Normalizes training batches using these dynamic stats.
    """
    
    def __init__(self, hparams):
        self.hparams = hparams
        self.training_files = hparams.training_files
        self.subset_size = getattr(hparams, 'dynamic_stats_subset_size', 10) # Number of utterances per speaker for stats
        
        # Determine feature paths suffix logic
        # Usually we just need to know where features are. 
        # The data loader parses training_files list.
        self.file_list_by_speaker = defaultdict(list)
        self._load_file_list()
        
        self.current_stats = {} # {speaker_id: {'energy': {'mean': ..., 'std': ...}, 'pitch': ..., 'spk_emb': ...}}
        
        # Initial refresh
        self.refresh_stats()
        
    def _load_file_list(self):
        with open(self.training_files, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('|') for line in f]
            
        for line in lines:
            features_dir = line[0]
            feature_file = line[1]
            speaker_id = int(line[2])
            
            # Store full paths needed for loading
            # We need paths to .frames_nrg, .frames_f0, .spk_emb.npy
            entry = {
                'energy': os.path.join(features_dir, f'{feature_file}.frames_nrg'),
                'pitch': os.path.join(features_dir, f'{feature_file}.frames_f0'),
                'spk_emb': os.path.join(features_dir, f'{feature_file}.spk_emb.npy')
            }
            self.file_list_by_speaker[speaker_id].append(entry)
            
    def refresh_stats(self):
        """Selects new random subsets and recomputes stats."""
        _logger.info("Refreshing dynamic speaker stats...")
        new_stats = {}
        
        for speaker_id, files in self.file_list_by_speaker.items():
            # Select subset
            # Select subset
            # Randomize size between 1 and min(len(files), self.subset_size)
            # This simulates variable reference data availability.
            max_k = min(len(files), self.subset_size)
            current_k = random.randint(1, max_k)
            subset = random.sample(files, current_k)
                
            # Accumulate values
            all_pitch = []
            all_energy = []
            all_embs = []
            
            for entry in subset:
                # Load pitch
                try:
                    with open(entry['pitch'], 'r') as f:
                        p = np.array([float(x.strip()) for x in f.readlines()])
                        all_pitch.extend(p[p > 0]) # Voice only
                except Exception as e:
                    _logger.warning(f"Error loading pitch {entry['pitch']}: {e}")
                    
                # Load energy
                try:
                    with open(entry['energy'], 'r') as f:
                        e = np.array([float(x.strip()) for x in f.readlines()])
                        all_energy.extend(e[e > 0]) # Non-zero only
                except:
                     pass
                     
                # Load Embedding
                try:
                    if os.path.exists(entry['spk_emb']):
                        emb = np.load(entry['spk_emb'])
                        all_embs.append(emb)
                except:
                    pass
            
            # Compute Stats
            if len(all_pitch) > 0:
                pitch_arr = np.array(all_pitch)
                p_mean = float(np.mean(pitch_arr))
                p_std = float(np.std(pitch_arr))
                if p_std == 0: p_std = 1.0 # Safety
            else:
                p_mean, p_std = 0.0, 1.0
                
            if len(all_energy) > 0:
                energy_arr = np.array(all_energy)
                e_mean = float(np.mean(energy_arr))
                e_std = float(np.std(energy_arr))
                if e_std == 0: e_std = 1.0
            else:
                e_mean, e_std = 0.0, 1.0
                
            # Average Embedding
            if len(all_embs) > 0:
                avg_emb = np.mean(np.array(all_embs), axis=0) # (192,)
            else:
                avg_emb = np.zeros(192) # Fallback
                
            new_stats[speaker_id] = {
                'pitch': {'mean': p_mean, 'std': p_std},
                'energy': {'mean': e_mean, 'std': e_std},
                'spk_emb': torch.from_numpy(avg_emb).float()
            }
            _logger.info(f"Refreshed Stats for Speaker {speaker_id}:")
            _logger.info(f"  Subset size: {len(subset)} files")
            _logger.info(f"  Pitch: mean={p_mean:.4f}, std={p_std:.4f}")
            _logger.info(f"  Energy: mean={e_mean:.4f}, std={e_std:.4f}")
            _logger.info(f"  Avg Spk Emb Shape: {avg_emb.shape}")
            
        self.current_stats = new_stats
        
    def process_batch(self, inputs, device):
        """
        Normalizes the raw pitch/energy in inputs using current stats 
        and replaces/injects the averaged speaker embedding.
        
        inputs tuple: 
        (symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths,
         frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids, feature_dirs, feature_files, spk_embs)
        """
        # Unpack
        symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths, \
            frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids, spk_embs = inputs
            
        feature_dirs, feature_files = None, None # Not available in processed inputs
            
        # We need to construct new tensors for normalized values and embeddings
        # We process item by item or vectorized if possible. 
        # Since stats are per speaker, we can iterate over unique speakers in batch.
        
        # Clone to avoid modifying original if needed, but in-place is fine for training
        frames_energy_norm = frames_energy.clone()
        frames_pitch_norm = frames_pitch.clone()
        # symbols_energy/pitch are derived from frames in preprocessing. 
        # Since we load raw, they should be raw too. We need to normalize them.
        symbols_energy_norm = symbols_energy.clone()
        symbols_pitch_norm = symbols_pitch.clone()
        
        # New embeddings tensor
        # shape (B, emb_dim)
        # We ignore the `spk_embs` coming from dataloader (which is per-utterance) 
        # and use the `avg_emb` from stats, OR we can use the one from dataloader if we want.
        # Requirement: "extract speaker embeddings from that random subset ... simulate test time"
        # So we MUST use the averaged embedding from the stats subset.
        
        # Get standard sizing from first speaker
        dummy_emb = next(iter(self.current_stats.values()))['spk_emb']
        avg_spk_embs = torch.zeros(len(speaker_ids), dummy_emb.shape[0]).to(device)
        
        # Unique speakers in batch
        unique_spks = torch.unique(speaker_ids)
        
        for spk_id in unique_spks:
            sid = spk_id.item()
            if sid not in self.current_stats:
                continue
                
            stats = self.current_stats[sid]
            mask = (speaker_ids == spk_id)
            
            # Normalize Energy
            # raw input has 0 for silence. 
            # (val - mean) / std. Silence remains 0? 
            # In data_loader.py: 
            # zero_idxs = np.where(energies == 0.)[0]
            # energies -= mean ...
            # energies[zero_idxs] = 0.
            
            # Frames Energy
            vals = frames_energy_norm[mask]
            zero_mask = (vals == 0.)
            vals = (vals - stats['energy']['mean']) / stats['energy']['std']
            vals[zero_mask] = 0.
            frames_energy_norm[mask] = vals
            
            # Symbols Energy (same stats)
            vals = symbols_energy_norm[mask]
            zero_mask = (vals == 0.)
            vals = (vals - stats['energy']['mean']) / stats['energy']['std']
            vals[zero_mask] = 0.
            symbols_energy_norm[mask] = vals
            
            # Pitch (Log-scale in file? Yes, extract_features says log pitch)
            # So just normalize
            vals = frames_pitch_norm[mask]
            zero_mask = (vals == 0.)
            vals = (vals - stats['pitch']['mean']) / stats['pitch']['std']
            vals[zero_mask] = 0.
            frames_pitch_norm[mask] = vals
            
            vals = symbols_pitch_norm[mask]
            zero_mask = (vals == 0.)
            vals = (vals - stats['pitch']['mean']) / stats['pitch']['std']
            vals[zero_mask] = 0.
            symbols_pitch_norm[mask] = vals
            
            # Assign Avg Embedding
            # stats['spk_emb'] is a tensor.
            emb = stats['spk_emb'].to(device)
            # Expand to mask size
            # avg_spk_embs[mask] = emb # Broadcasting works?
            # mask has M true elements. emb is (D,). Target (M, D).
            avg_spk_embs[mask] = emb
            
        # Re-pack inputs
        # Note: model.parse_batch unpacks 14 items.
        # We need to return the expected tuple format for model()
        # Model forward takes (..., spk_embs)
        
        # Wait, train loop calls parse_batch then model(inputs).
        # We intercept inputs.
        
        new_inputs = (symbols, durations_float, durations_int, symbols_energy_norm, symbols_pitch_norm, input_lengths,
                      frames_energy_norm, frames_pitch_norm, mel_specs, output_lengths, speaker_ids, avg_spk_embs)
                      
        return new_inputs
