import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F


class DaftExprtDataLoader(Dataset):
    ''' Load PyTorch Data Set
        1) load features, symbols and speaker ID
        2) convert symbols to sequence of one-hot vectors
    '''
    def __init__(self, data_file, hparams, shuffle=True):
        # check data file exists and extract lines
        assert(os.path.isfile(data_file))
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.data = [line.strip().split(sep='|') for line in lines]
        self.hparams = hparams
        
        # shuffle
        if shuffle:
            random.seed(hparams.seed)
            random.shuffle(self.data)
    
    def get_mel_spec(self, mel_spec):
        ''' Extract PyTorch float tensor from .npy mel-spec file
        '''
        # transform to PyTorch tensor and check size
        mel_spec = torch.from_numpy(np.load(mel_spec))
        assert(mel_spec.size(0) == self.hparams.n_mel_channels)
        
        return mel_spec
    
    def get_symbols_and_durations(self, markers):
        ''' Extract PyTorch int tensor from an input symbols sequence
            Extract PyTorch float and int duration for each symbol
        '''
        # initialize variables
        symbols, durations_float, durations_int = [], [], []
        
        # read lines of markers file
        with open(markers, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        markers = [line.strip().split(sep='\t') for line in lines]
        
        # iterate over markers
        for marker in markers:
            begin, end, int_dur, symbol, _, _ = marker
            symbols.append(self.hparams.symbols.index(symbol))
            durations_float.append(float(end) - float(begin))
            durations_int.append(int(int_dur))
        
        # convert lists to PyTorch tensors
        symbols = torch.IntTensor(symbols)
        durations_float = torch.FloatTensor(durations_float)
        durations_int = torch.IntTensor(durations_int)
        
        return symbols, durations_float, durations_int
    
    def get_energies(self, energies, speaker_id, normalize=True):
        ''' Extract standardized PyTorch float tensor for energies
        '''
        # read energy lines
        with open(energies, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        energies = np.array([float(line.strip()) for line in lines])
        # standardize energies based on speaker stats
        if normalize:
            zero_idxs = np.where(energies == 0.)[0]
            energies -= self.hparams.stats[f'spk {speaker_id}']['energy']['mean']
            energies /= self.hparams.stats[f'spk {speaker_id}']['energy']['std']
            energies[zero_idxs] = 0.
        # convert to PyTorch float tensor
        energies = torch.FloatTensor(energies)
        
        return energies
    
    def get_pitch(self, pitch, speaker_id, normalize=True):
        ''' Extract standardized PyTorch float tensor for pitch
        '''
        # read pitch lines
        with open(pitch, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        pitch = np.array([float(line.strip()) for line in lines])
        # standardize voiced pitch based on speaker stats
        if normalize:
            zero_idxs = np.where(pitch == 0.)[0]
            pitch -= self.hparams.stats[f'spk {speaker_id}']['pitch']['mean']
            pitch /= self.hparams.stats[f'spk {speaker_id}']['pitch']['std']
            pitch[zero_idxs] = 0.
        # convert to PyTorch float tensor
        pitch = torch.FloatTensor(pitch)
        
        return pitch
    
    def get_data(self, data):
        ''' Extract features, symbols and speaker ID
        '''
        # get mel-spec path, markers path, pitch path and speaker ID
        features_dir = data[0]
        feature_file = data[1]
        speaker_id = int(data[2])
        
        mel_spec = os.path.join(features_dir, f'{feature_file}.npy')
        markers = os.path.join(features_dir, f'{feature_file}.markers')
        symbols_energy = os.path.join(features_dir, f'{feature_file}.symbols_nrg')
        frames_energy = os.path.join(features_dir, f'{feature_file}.frames_nrg')
        symbols_pitch = os.path.join(features_dir, f'{feature_file}.symbols_f0')
        frames_pitch = os.path.join(features_dir, f'{feature_file}.frames_f0')
        
        # extract data
        mel_spec = self.get_mel_spec(mel_spec)
        symbols, durations_float, durations_int = self.get_symbols_and_durations(markers)
        symbols_energy = self.get_energies(symbols_energy, speaker_id)
        frames_energy = self.get_energies(frames_energy, speaker_id, normalize=False)
        symbols_pitch = self.get_pitch(symbols_pitch, speaker_id)
        frames_pitch = self.get_pitch(frames_pitch, speaker_id, normalize=False)
        
        # check everything is correct with sizes
        if len(symbols_energy) != len(symbols):
            print(f"ERROR: Energy/Symbols mismatch in {features_dir}/{feature_file}")
            print(f"Energy len: {len(symbols_energy)}, Symbols len: {len(symbols)}")
        assert(len(symbols_energy) == len(symbols))
        
        if len(symbols_pitch) != len(symbols):
            print(f"ERROR: Pitch/Symbols mismatch in {features_dir}/{feature_file}")
            print(f"Pitch len: {len(symbols_pitch)}, Symbols len: {len(symbols)}")
        assert(len(symbols_pitch) == len(symbols))
        
        if len(frames_energy) != mel_spec.size(1):
            print(f"ERROR: Frames Energy/Mel mismatch in {features_dir}/{feature_file}")
            print(f"Frames Energy len: {len(frames_energy)}, Mel len: {mel_spec.size(1)}")
        assert(len(frames_energy) == mel_spec.size(1))
        
        if len(frames_pitch) != mel_spec.size(1):
            print(f"ERROR: Frames Pitch/Mel mismatch in {features_dir}/{feature_file}")
            print(f"Frames Pitch len: {len(frames_pitch)}, Mel len: {mel_spec.size(1)}")
        assert(len(frames_pitch) == mel_spec.size(1))
        
        if torch.sum(durations_int) != mel_spec.size(1):
            print(f"ERROR: Duration/Mel mismatch in {features_dir}/{feature_file}")
            print(f"Duration sum: {torch.sum(durations_int)}, Mel len: {mel_spec.size(1)}")
        assert(torch.sum(durations_int) == mel_spec.size(1))
        
        # Apply augmentation if enabled
        if hasattr(self.hparams, 'aug_prob') and self.hparams.aug_prob > 0:
            if random.random() < self.hparams.aug_prob:
                symbols, durations_float, durations_int, symbols_energy, symbols_pitch, \
                frames_energy, frames_pitch, mel_spec = self._augment_data(
                    symbols, durations_float, durations_int, symbols_energy, symbols_pitch,
                    frames_energy, frames_pitch, mel_spec, speaker_id
                )

        return symbols, durations_float, durations_int, symbols_energy, symbols_pitch, \
            frames_energy, frames_pitch, mel_spec, speaker_id, features_dir, feature_file
    
    def _augment_data(self, symbols, durations_float, durations_int, symbols_energy, symbols_pitch,
                      frames_energy, frames_pitch, mel_spec, speaker_id):
        ''' Apply random prosody augmentation
        '''
        # 1. Pitch Shift (Bin-First Logic)
        if hasattr(self.hparams, 'max_mel_shift') and self.hparams.max_mel_shift > 0:
            # Randomly select integer bin shift
            shift_bins = random.randint(-self.hparams.max_mel_shift, self.hparams.max_mel_shift)
            
            if shift_bins != 0:
                # Calculate approximate semitone shift
                # Approximation: 1 Mel Bin ~= 0.7 Semitones
                semitones = shift_bins * 0.7
                
                pitch_std = self.hparams.stats[f'spk {speaker_id}']['pitch']['std']
                if pitch_std > 0:
                    shift_val = (semitones / 12.0) * np.log(2) / pitch_std
                    
                    # Apply to symbols_pitch (only voiced)
                    mask = (symbols_pitch != 0)
                    symbols_pitch[mask] += shift_val
                    
                    # Apply to frames_pitch (only voiced)
                    mask = (frames_pitch != 0)
                    frames_pitch[mask] += shift_val
                
                # Mel Shifting (Shift with Padding)
                # mel_spec shape: (80, T)
                # We shift along dim 0 (frequency)
                # Use min_clipping value for padding (log domain)
                min_val = np.log(self.hparams.min_clipping)
                
                if shift_bins > 0:
                    # Shift UP: Pad bottom, Crop top
                    # New[i] = Old[i - shift]
                    # New[0..shift] = min_val
                    shifted_mel = torch.roll(mel_spec, shift_bins, dims=0)
                    shifted_mel[:shift_bins, :] = min_val
                    mel_spec = shifted_mel
                else:
                    # Shift DOWN: Pad top, Crop bottom
                    # New[i] = Old[i - shift] (shift is neg, so i + abs(shift))
                    # New[end+shift..end] = min_val
                    shifted_mel = torch.roll(mel_spec, shift_bins, dims=0)
                    shifted_mel[shift_bins:, :] = min_val
                    mel_spec = shifted_mel 

        # 2. Energy Scale
        if hasattr(self.hparams, 'energy_scale_min') and hasattr(self.hparams, 'energy_scale_max'):
            scale = random.uniform(self.hparams.energy_scale_min, self.hparams.energy_scale_max)
            # Energy is normalized. new_val = (val * scale - mean) / std ? 
            # No, energy features are usually L2 norm or similar.
            # If we scale signal by factor A, energy scales by A (or A^2 depending on def).
            # Let's assume linear scaling of amplitude.
            # Mel spec is log(magnitude). So log(A * mag) = log(A) + log(mag).
            # So we add log(scale) to mel-spec.
            
            log_scale = np.log(scale)
            mel_spec = mel_spec + log_scale
            
            # Update energy features.
            # frames_energy is (energy - mean) / std.
            # We need to know if 'energy' is log-energy or linear energy.
            # Usually it's L2 norm of frame.
            # If we scale audio by A, L2 norm scales by A.
            # So we multiply the raw energy by A.
            # new_norm = (raw * scale - mean) / std = (raw - mean + mean) * scale / std - mean/std
            # = (norm * std + mean) * scale / std - mean/std
            # = norm * scale + mean/std * (scale - 1)
            
            energy_mean = self.hparams.stats[f'spk {speaker_id}']['energy']['mean']
            energy_std = self.hparams.stats[f'spk {speaker_id}']['energy']['std']
            
            if energy_std > 0:
                # Apply to symbols_energy
                mask = (symbols_energy != 0)
                symbols_energy[mask] = symbols_energy[mask] * scale + (energy_mean / energy_std) * (scale - 1)
                
                # Apply to frames_energy
                mask = (frames_energy != 0)
                frames_energy[mask] = frames_energy[mask] * scale + (energy_mean / energy_std) * (scale - 1)

        # 3. Time Stretch
        if hasattr(self.hparams, 'time_stretch_min') and hasattr(self.hparams, 'time_stretch_max'):
            stretch = random.uniform(self.hparams.time_stretch_min, self.hparams.time_stretch_max)
            if stretch != 1.0:
                # Stretch durations
                durations_float = durations_float * stretch
                
                # Re-calculate int durations
                # We need to ensure sum matches new mel-spec length
                # New mel length = old * stretch
                new_mel_len = int(mel_spec.size(1) * stretch)
                
                # Interpolate mel-spec
                # mel_spec: (80, T) -> (1, 80, T)
                mel_spec = F.interpolate(mel_spec.unsqueeze(0), size=new_mel_len, mode='linear', align_corners=False).squeeze(0)
                
                # Interpolate frames_energy and frames_pitch
                # (T) -> (1, 1, T)
                frames_energy = F.interpolate(frames_energy.unsqueeze(0).unsqueeze(0), size=new_mel_len, mode='linear', align_corners=False).squeeze(0).squeeze(0)
                frames_pitch = F.interpolate(frames_pitch.unsqueeze(0).unsqueeze(0), size=new_mel_len, mode='linear', align_corners=False).squeeze(0).squeeze(0)
                
                # Adjust durations_int to match new_mel_len
                # We can use the same logic as in model.py or simple rounding and fixing the diff
                durations_int = torch.round(durations_float).long()
                diff = new_mel_len - torch.sum(durations_int).item()
                if diff != 0:
                    # Add/subtract diff to the largest duration to minimize distortion
                    argmax = torch.argmax(durations_int)
                    durations_int[argmax] += diff
                    # Ensure no negative/zero if possible (though argmax usually safe)
                    if durations_int[argmax] < 1:
                         durations_int[argmax] = 1
                         # If we still have mismatch, just force resize mel? No, simpler to fix duration.
                         # Re-check sum
                         diff2 = new_mel_len - torch.sum(durations_int).item()
                         if diff2 != 0:
                             # Just trim/pad mel spec to match durations (easier)
                             target_len = torch.sum(durations_int).item()
                             if target_len > new_mel_len:
                                 # Pad mel
                                 pad_amt = target_len - new_mel_len
                                 mel_spec = F.pad(mel_spec, (0, pad_amt))
                                 frames_energy = F.pad(frames_energy, (0, pad_amt))
                                 frames_pitch = F.pad(frames_pitch, (0, pad_amt))
                             elif target_len < new_mel_len:
                                 # Trim mel
                                 mel_spec = mel_spec[:, :target_len]
                                 frames_energy = frames_energy[:target_len]
                                 frames_pitch = frames_pitch[:target_len]

        return symbols, durations_float, durations_int, symbols_energy, symbols_pitch, \
               frames_energy, frames_pitch, mel_spec
    
    def __getitem__(self, index):
        return self.get_data(self.data[index])

    def __len__(self):
        return len(self.data)


class DaftExprtDataCollate():
    ''' Zero-pads model inputs and targets
    '''
    def __init__(self, hparams):
        self.hparams = hparams
    
    def __call__(self, batch):
        ''' Collate training batch

        :param batch:   [[symbols, durations_float, durations_int, symbols_energy, symbols_pitch,
                          frames_energy, frames_pitch, mel_spec, speaker_id, features_dir, feature_file], ...]

        :return: collated batch of training samples
        '''
        # find symbols sequence max length
        input_lengths, ids_sorted_decreasing = \
            torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
        max_input_len = input_lengths[0]
        
        # right zero-pad sequences to max input length
        symbols = torch.LongTensor(len(batch), max_input_len).zero_()
        durations_float = torch.FloatTensor(len(batch), max_input_len).zero_()
        durations_int = torch.LongTensor(len(batch), max_input_len).zero_()
        symbols_energy = torch.FloatTensor(len(batch), max_input_len).zero_()
        symbols_pitch = torch.FloatTensor(len(batch), max_input_len).zero_()
        speaker_ids = torch.LongTensor(len(batch))
        
        for i in range(len(ids_sorted_decreasing)):
            # extract batch sequences
            symbols_seq = batch[ids_sorted_decreasing[i]][0]
            dur_float_seq = batch[ids_sorted_decreasing[i]][1]
            dur_int_seq = batch[ids_sorted_decreasing[i]][2]
            symbols_energy_seq = batch[ids_sorted_decreasing[i]][3]
            symbols_pitch_seq = batch[ids_sorted_decreasing[i]][4]
            # fill padded arrays
            symbols[i, :symbols_seq.size(0)] = symbols_seq
            durations_float[i, :dur_float_seq.size(0)] = dur_float_seq
            durations_int[i, :dur_int_seq.size(0)] = dur_int_seq
            symbols_energy[i, :symbols_energy_seq.size(0)] = symbols_energy_seq
            symbols_pitch[i, :symbols_pitch_seq.size(0)] = symbols_pitch_seq
            # add corresponding speaker ID
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][8]
        
        # find mel-spec max length
        max_output_len = max([x[7].size(1) for x in batch])
        
        # right zero-pad mel-specs to max output length
        frames_energy = torch.FloatTensor(len(batch), max_output_len).zero_()
        frames_pitch = torch.FloatTensor(len(batch), max_output_len).zero_()
        mel_specs = torch.FloatTensor(len(batch), self.hparams.n_mel_channels, max_output_len).zero_()
        output_lengths = torch.LongTensor(len(batch))
        
        for i in range(len(ids_sorted_decreasing)):
            # extract batch sequences
            frames_energy_seq = batch[ids_sorted_decreasing[i]][5]
            frames_pitch_seq = batch[ids_sorted_decreasing[i]][6]
            mel_spec = batch[ids_sorted_decreasing[i]][7]
            # fill padded arrays
            frames_energy[i, :frames_energy_seq.size(0)] = frames_energy_seq
            frames_pitch[i, :frames_pitch_seq.size(0)] = frames_pitch_seq
            mel_specs[i, :, :mel_spec.size(1)] = mel_spec
            output_lengths[i] = mel_spec.size(1)
        
        # store file identification
        # only used in fine_tune.py script
        feature_dirs, feature_files = [], []
        for i in range(len(ids_sorted_decreasing)):
            feature_dirs.append(batch[ids_sorted_decreasing[i]][9])
            feature_files.append(batch[ids_sorted_decreasing[i]][10])
        
        return symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths, \
            frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids, feature_dirs, feature_files


def prepare_data_loaders(hparams, num_workers=1, drop_last=True):
    ''' Initialize train and validation Data Loaders

    :param hparams:             hyper-parameters used for training
    :param num_workers:         number of workers involved in the Data Loader

    :return: Data Loaders for train and validation sets
    '''
    # get data and collate function ready
    train_set = DaftExprtDataLoader(hparams.training_files, hparams)
    val_set = DaftExprtDataLoader(hparams.validation_files, hparams)
    collate_fn = DaftExprtDataCollate(hparams)
    
    # get number of training examples
    nb_training_examples = len(train_set)
    
    # use distributed sampler if we use distributed training
    if hparams.multiprocessing_distributed:
        train_sampler = DistributedSampler(train_set, shuffle=False)
    else:
        train_sampler = None
    
    # build training and validation data loaders
    # drop_last=True because we shuffle data set at each epoch
    train_loader = DataLoader(train_set, num_workers=num_workers, shuffle=(train_sampler is None), sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=True, drop_last=drop_last, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, num_workers=num_workers, shuffle=False, batch_size=hparams.batch_size,
                            pin_memory=True, drop_last=False, collate_fn=collate_fn)
    
    return train_loader, train_sampler, val_loader, nb_training_examples
