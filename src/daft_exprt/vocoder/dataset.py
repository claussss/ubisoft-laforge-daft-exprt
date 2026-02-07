"""Dataset and mel-spectrogram extraction for HiFi-GAN fine-tuning.

Provides a PyTorch Dataset that loads (predicted_mel, ground_truth_audio) pairs
produced by Daft-Exprt's ``fine_tune.py``, plus the mel-spectrogram function
used to compute the L1 reconstruction loss during training.
"""

import math
import os
import random

import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read as read_wav


MAX_WAV_VALUE = 32768.0

# Module-level caches (same pattern as official HiFi-GAN meldataset.py)
_mel_basis = {}
_hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size,
                    fmin, fmax, center=False):
    """Compute log-mel spectrogram, matching official HiFi-GAN extraction.

    Used during training to compute the L1 reconstruction loss between
    generated and ground-truth audio.  The ``fmax`` for loss computation
    is typically ``None`` (full bandwidth up to Nyquist), while the input
    conditioning mel uses ``fmax=8000``.
    """
    if torch.min(y) < -1.0:
        print('mel_spectrogram: min value is', torch.min(y).item())
    if torch.max(y) > 1.0:
        print('mel_spectrogram: max value is', torch.max(y).item())

    global _mel_basis, _hann_window

    fmax_key = str(fmax) + '_' + str(y.device)
    if fmax_key not in _mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
                             fmin=fmin, fmax=fmax)
        _mel_basis[fmax_key] = torch.from_numpy(mel).float().to(y.device)

    win_key = str(y.device)
    if win_key not in _hann_window:
        _hann_window[win_key] = torch.hann_window(win_size).to(y.device)

    # Reflect-pad to emulate center=True while keeping center=False
    pad = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
                      window=_hann_window[win_key], center=center,
                      pad_mode='reflect', normalized=False, onesided=True,
                      return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(_mel_basis[fmax_key], spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))
    return spec


def _find_pairs(root_dir):
    """Walk *root_dir* recursively and return sorted list of (mel_path, wav_path) pairs."""
    pairs = []
    for dirpath, _dirs, files in os.walk(root_dir):
        wav_files = {f for f in files if f.endswith('.wav')}
        for wf in sorted(wav_files):
            stem = os.path.splitext(wf)[0]
            mel_path = os.path.join(dirpath, stem + '.npy')
            if os.path.isfile(mel_path):
                pairs.append((mel_path, os.path.join(dirpath, wf)))
    pairs.sort(key=lambda p: p[1])
    return pairs


class HiFiGANFinetuneDataset(torch.utils.data.Dataset):
    """Loads predicted-mel / ground-truth-audio pairs for HiFi-GAN fine-tuning.

    Expects a directory tree where each ``.npy`` file has a matching ``.wav``
    file with the same stem (the layout produced by ``fine_tune.py``).

    During training (``split=True``), random segments of ``segment_size``
    audio samples are cropped.  The corresponding mel frames are sliced
    to ``ceil(segment_size / hop_size)`` frames.
    """

    def __init__(self, root_dir, segment_size, n_fft, num_mels, hop_size,
                 win_size, sampling_rate, fmin, fmax, split=True,
                 shuffle=True, fmax_loss=None, device=None):
        self.pairs = _find_pairs(root_dir)
        if not self.pairs:
            raise RuntimeError(
                f'No mel/wav pairs found in {root_dir}. '
                'Expected .npy and .wav files with matching stems.')
        if shuffle:
            random.seed(1234)
            random.shuffle(self.pairs)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.device = device

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        mel_path, wav_path = self.pairs[index]

        # --- audio --------------------------------------------------------
        sr, audio = read_wav(wav_path)
        if sr != self.sampling_rate:
            raise ValueError(
                f'{wav_path}: sample rate {sr} != expected {self.sampling_rate}')
        audio = audio.astype(np.float32) / MAX_WAV_VALUE
        audio = torch.FloatTensor(audio).unsqueeze(0)  # (1, T)

        # --- mel (predicted by TTS) ---------------------------------------
        mel = np.load(mel_path)
        mel = torch.from_numpy(mel).float()
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # (1, n_mels, T_mel)

        # --- segment cropping ---------------------------------------------
        if self.split:
            frames_per_seg = math.ceil(self.segment_size / self.hop_size)

            if audio.size(1) >= self.segment_size:
                mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                audio = audio[:, mel_start * self.hop_size:
                              (mel_start + frames_per_seg) * self.hop_size]
            else:
                mel = torch.nn.functional.pad(
                    mel, (0, frames_per_seg - mel.size(2)), 'constant')
                audio = torch.nn.functional.pad(
                    audio, (0, self.segment_size - audio.size(1)), 'constant')

        # --- mel for L1 loss (computed from GT audio) ---------------------
        mel_loss = mel_spectrogram(
            audio, self.n_fft, self.num_mels, self.sampling_rate,
            self.hop_size, self.win_size, self.fmin, self.fmax_loss,
            center=False)

        return mel.squeeze(0), audio.squeeze(0), wav_path, mel_loss.squeeze(0)
