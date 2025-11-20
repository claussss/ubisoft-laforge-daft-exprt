#!/usr/bin/env python3

"""Verify a generated mel-spectrogram by converting it to audio using HiFi-GAN.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

FILE_ROOT = Path(__file__).resolve()
PROJECT_ROOT = FILE_ROOT.parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
sys.path.append(str(SRC_ROOT))
os.environ.setdefault('PYTHONPATH', str(SRC_ROOT))

from daft_exprt.vocoder import load_hifigan_vocoder

def main():
    parser = argparse.ArgumentParser(description='Verify mel-spectrogram by generating audio.')
    parser.add_argument('--mel', required=True, help='Path to the mel-spectrogram .npy file.')
    parser.add_argument('--out-wav', required=True, help='Where to store the reconstructed waveform.')
    parser.add_argument('--vocoder-checkpoint', default='', help='Optional path to a HiFi-GAN generator checkpoint.')
    parser.add_argument('--device', default='', help='Torch device for the vocoder (defaults to cuda if available).')
    parser.add_argument('--sampling-rate', type=int, default=22050, help='Sampling rate.')
    args = parser.parse_args()

    print(f"Loading mel-spectrogram from {args.mel}...")
    mel_spec = np.load(args.mel)
    print(f"Mel-spectrogram shape: {mel_spec.shape}")

    print("Loading HiFi-GAN vocoder...")
    vocoder = load_hifigan_vocoder(
        checkpoint_path=args.vocoder_checkpoint or None,
        device=args.device or None,
    )
    
    print("Generating audio...")
    audio = vocoder.infer(mel_spec)
    audio_int16 = (audio * 32767.5).clip(min=-32768, max=32767).astype(np.int16)
    
    print(f"Saving audio to {args.out_wav}...")
    wavfile.write(args.out_wav, args.sampling_rate, audio_int16)
    print("Done!")

if __name__ == '__main__':
    main()
