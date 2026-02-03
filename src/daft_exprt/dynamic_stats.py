"""On-the-fly prosody normalization during training.

Maintains a random subset of training files per speaker, periodically refreshes
mean/std of pitch and energy and average speaker embedding from that subset,
and normalizes training batches using these dynamic stats (simulating variable
reference data at test time).
"""

import logging
import os
import random
from collections import defaultdict

import numpy as np
import torch

_logger = logging.getLogger(__name__)


class DynamicSpeakerStatsManager:
    """
    Manages dynamic speaker statistics and embeddings during training.

    Maintains a subset of "support set" files per speaker. Periodically refreshes
    this subset, computes mean/std of pitch/energy and average speaker embedding,
    and normalizes training batches using these dynamic stats.
    """

    def __init__(self, hparams):
        self.hparams = hparams
        self.training_files = hparams.training_files
        self.subset_size = getattr(hparams, 'dynamic_stats_subset_size', 10)
        self.emb_dim = getattr(hparams, 'external_emb_dim', 192)

        self.file_list_by_speaker = defaultdict(list)
        self._load_file_list()

        self.current_stats = {}

        self.refresh_stats()

    def _load_file_list(self):
        with open(self.training_files, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('|') for line in f]

        for line in lines:
            if len(line) < 3:
                continue
            features_dir = line[0]
            feature_file = line[1]
            speaker_id = int(line[2])

            entry = {
                'energy': os.path.join(features_dir, f'{feature_file}.frames_nrg'),
                'pitch': os.path.join(features_dir, f'{feature_file}.frames_f0'),
                'spk_emb': os.path.join(features_dir, f'{feature_file}.spk_emb.npy'),
            }
            self.file_list_by_speaker[speaker_id].append(entry)

    def refresh_stats(self):
        """Select new random subsets per speaker and recompute stats."""
        _logger.info("Refreshing dynamic speaker stats...")
        new_stats = {}

        for speaker_id, files in self.file_list_by_speaker.items():
            max_k = min(len(files), self.subset_size)
            current_k = random.randint(1, max_k)
            subset = random.sample(files, current_k)

            all_pitch = []
            all_energy = []
            all_embs = []

            for entry in subset:
                try:
                    with open(entry['pitch'], 'r', encoding='utf-8') as f:
                        p = np.array([float(x.strip()) for x in f.readlines()])
                        all_pitch.extend(p[p > 0])
                except Exception as e:
                    _logger.warning("Error loading pitch %s: %s", entry['pitch'], e)

                try:
                    with open(entry['energy'], 'r', encoding='utf-8') as f:
                        e = np.array([float(x.strip()) for x in f.readlines()])
                        all_energy.extend(e[e > 0])
                except Exception as e:
                    _logger.warning("Error loading energy %s: %s", entry['energy'], e)

                try:
                    if os.path.exists(entry['spk_emb']):
                        emb = np.load(entry['spk_emb'])
                        all_embs.append(emb)
                except Exception:
                    pass

            if len(all_pitch) > 0:
                pitch_arr = np.array(all_pitch)
                p_mean = float(np.mean(pitch_arr))
                p_std = float(np.std(pitch_arr))
                if p_std == 0:
                    p_std = 1.0
            else:
                p_mean, p_std = 0.0, 1.0

            if len(all_energy) > 0:
                energy_arr = np.array(all_energy)
                e_mean = float(np.mean(energy_arr))
                e_std = float(np.std(energy_arr))
                if e_std == 0:
                    e_std = 1.0
            else:
                e_mean, e_std = 0.0, 1.0

            if len(all_embs) > 0:
                avg_emb = np.mean(np.array(all_embs), axis=0)
            else:
                avg_emb = np.zeros(self.emb_dim)

            new_stats[speaker_id] = {
                'pitch': {'mean': p_mean, 'std': p_std},
                'energy': {'mean': e_mean, 'std': e_std},
                'spk_emb': torch.from_numpy(avg_emb).float(),
            }
            _logger.info(
                "Refreshed stats for speaker %s: subset=%s, pitch mean=%.4f std=%.4f, energy mean=%.4f std=%.4f",
                speaker_id, len(subset), p_mean, p_std, e_mean, e_std,
            )

        self.current_stats = new_stats

    def process_batch(self, inputs, device):
        """
        Normalize pitch/energy in inputs using current_stats and replace
        speaker embeddings with the averaged embedding from the stats subset.

        inputs: tuple of 12 elements from model.parse_batch (symbols, durations_float,
                durations_int, symbols_energy, symbols_pitch, input_lengths,
                frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids, spk_embs)
        """
        (symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths,
         frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids, spk_embs) = inputs

        frames_energy_norm = frames_energy.clone()
        frames_pitch_norm = frames_pitch.clone()
        symbols_energy_norm = symbols_energy.clone()
        symbols_pitch_norm = symbols_pitch.clone()

        dummy_emb = next(iter(self.current_stats.values()))['spk_emb']
        avg_spk_embs = torch.zeros(len(speaker_ids), dummy_emb.shape[0]).to(device)

        unique_spks = torch.unique(speaker_ids)

        for spk_id in unique_spks:
            sid = spk_id.item()
            if sid not in self.current_stats:
                continue

            stats = self.current_stats[sid]
            mask = (speaker_ids == spk_id)

            # Energy: preserve zeros (silence)
            vals = frames_energy_norm[mask]
            zero_mask = (vals == 0.0)
            vals = (vals - stats['energy']['mean']) / stats['energy']['std']
            vals[zero_mask] = 0.0
            frames_energy_norm[mask] = vals

            vals = symbols_energy_norm[mask]
            zero_mask = (vals == 0.0)
            vals = (vals - stats['energy']['mean']) / stats['energy']['std']
            vals[zero_mask] = 0.0
            symbols_energy_norm[mask] = vals

            # Pitch: preserve zeros
            vals = frames_pitch_norm[mask]
            zero_mask = (vals == 0.0)
            vals = (vals - stats['pitch']['mean']) / stats['pitch']['std']
            vals[zero_mask] = 0.0
            frames_pitch_norm[mask] = vals

            vals = symbols_pitch_norm[mask]
            zero_mask = (vals == 0.0)
            vals = (vals - stats['pitch']['mean']) / stats['pitch']['std']
            vals[zero_mask] = 0.0
            symbols_pitch_norm[mask] = vals

            emb = stats['spk_emb'].to(device)
            avg_spk_embs[mask] = emb

        new_inputs = (
            symbols, durations_float, durations_int, symbols_energy_norm, symbols_pitch_norm,
            input_lengths, frames_energy_norm, frames_pitch_norm, mel_specs, output_lengths,
            speaker_ids, avg_spk_embs,
        )
        return new_inputs
