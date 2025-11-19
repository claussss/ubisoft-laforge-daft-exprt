#!/usr/bin/env python3

"""Reconstruct a waveform from a ground-truth mel using the bundled HiFi-GAN vocoder.

This script helps validate whether the vocoder itself (versus mismatched mels)
is responsible for metallic / buzzing artifacts.
"""

import argparse
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from scipy.io import wavfile

FILE_ROOT = Path(__file__).resolve()
PROJECT_ROOT = FILE_ROOT.parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
sys.path.append(str(SRC_ROOT))
os.environ.setdefault('PYTHONPATH', str(SRC_ROOT))

from daft_exprt.vocoder import load_hifigan_vocoder  # noqa: E402


def hifigan_mel_spectrogram(
    wav,
    sampling_rate=22050,
    n_fft=1024,
    win_size=1024,
    hop_size=256,
    fmin=0.0,
    fmax=8000.0,
    n_mels=80,
    min_clipping=1e-5,
):
    """Compute a mel spectrogram using the canonical HiFi-GAN recipe (center=False + reflect pad)."""
    if wav.ndim != 1:
        raise ValueError('Expected a mono waveform.')
    wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
    pad = int((n_fft - hop_size) / 2)
    wav_tensor = F.pad(wav_tensor.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)
    wav_tensor = wav_tensor.squeeze(0)

    mel_filter_bank = librosa_mel_fn(
        sampling_rate,
        n_fft,
        n_mels,
        fmin,
        fmax,
    )
    mel_filter_bank = torch.from_numpy(mel_filter_bank).float()
    hann_window = torch.hann_window(win_size)

    spec = torch.stft(
        wav_tensor,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=False,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    mel_spec = torch.matmul(mel_filter_bank, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=min_clipping))
    return mel_spec.numpy()


def main():
    parser = argparse.ArgumentParser(description='Sanity-check HiFi-GAN by vocoding a reference mel.')
    parser.add_argument('--wav', required=True, help='Path to a reference waveform to analyze.')
    parser.add_argument('--out-wav', required=True, help='Where to store the reconstructed waveform.')
    parser.add_argument('--mel-out', default='', help='Optional path to save the extracted mel as .npy.')
    parser.add_argument('--vocoder-checkpoint', default='', help='Optional path to a HiFi-GAN generator checkpoint.')
    parser.add_argument('--device', default='', help='Torch device for the vocoder (defaults to cuda if available).')
    parser.add_argument('--sampling-rate', type=int, default=22050, help='Sampling rate to load/resample the WAV.')
    args = parser.parse_args()

    wav, _ = librosa.load(args.wav, sr=args.sampling_rate)
    wav = wav.astype(np.float32)

    mel_spec = hifigan_mel_spectrogram(
        wav,
        sampling_rate=args.sampling_rate,
    )

    vocoder = load_hifigan_vocoder(
        checkpoint_path=args.vocoder_checkpoint or None,
        device=args.device or None,
    )
    audio = vocoder.infer(mel_spec)
    audio_int16 = (audio * 32767.5).clip(min=-32768, max=32767).astype(np.int16)
    wavfile.write(args.out_wav, args.sampling_rate, audio_int16)

    if args.mel_out:
        np.save(args.mel_out, mel_spec)


if __name__ == '__main__':
    main()
