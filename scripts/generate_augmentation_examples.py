import argparse
import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
import torch.nn.functional as F

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from daft_exprt.hparams import HyperParams
from daft_exprt.extract_features import mel_spectrogram_HiFi, rescale_wav_to_float32
from daft_exprt.vocoder import load_hifigan_vocoder

def pitch_shift_bin_first(mel_spec, shift_bins, min_clipping=1e-5):
    """
    Apply Bin-First Pitch Shifting to a Mel-Spectrogram.
    mel_spec: (1, 80, T) or (80, T) torch tensor
    shift_bins: int, number of bins to shift
    """
    if shift_bins == 0:
        return mel_spec
    
    # Ensure (1, 80, T) for consistency if needed, but roll works on dims
    # We assume mel_spec is (80, T) or (1, 80, T)
    # We shift along the frequency dimension (dim -2)
    
    min_val = np.log(min_clipping)
    
    # Clone to avoid modifying original
    shifted = mel_spec.clone()
    
    # Roll
    shifted = torch.roll(shifted, shift_bins, dims=-2)
    
    # Pad
    if shift_bins > 0:
        # Shift UP: Pad bottom (low freq), Crop top (high freq) is implicit by roll but we need to silence rolled-around part
        # If (80, T), dim -2 is 0. If (1, 80, T), dim -2 is 1.
        if shifted.dim() == 2:
            shifted[:shift_bins, :] = min_val
        else:
            shifted[:, :shift_bins, :] = min_val
    else:
        # Shift DOWN: Pad top (high freq)
        if shifted.dim() == 2:
            shifted[shift_bins:, :] = min_val
        else:
            shifted[:, shift_bins:, :] = min_val
            
    return shifted

def time_stretch(mel_spec, factor):
    """
    Apply Time Stretching.
    mel_spec: (1, 80, T) torch tensor
    factor: float, <1 is fast, >1 is slow (duration multiplier)
    """
    # F.interpolate expects (B, C, T)
    if mel_spec.dim() == 2:
        mel_spec = mel_spec.unsqueeze(0)
    
    target_len = int(mel_spec.shape[-1] * factor)
    
    # Interpolate
    stretched = F.interpolate(mel_spec, size=target_len, mode='linear', align_corners=False)
    
    return stretched

def energy_scale(mel_spec, factor):
    """
    Apply Energy Scaling.
    mel_spec: (1, 80, T) torch tensor (log domain)
    factor: float, linear energy scale factor
    """
    # mel_spec is in log domain.
    # We want to scale the linear energy by factor.
    # log(energy * factor) = log(energy) + log(factor)
    # So we add log(factor) to the mel spec.
    
    return mel_spec + np.log(factor)

def main():
    parser = argparse.ArgumentParser(description='Generate augmentation examples')
    parser.add_argument('--input_wav', type=str, 
                        default=os.path.join(PROJECT_ROOT, 'scripts', 'style_bank', 'english', '0012_000567.wav'),
                        help='Input wav file')
    parser.add_argument('--output_dir', type=str, 
                        default='/home/yurii/Projects/LLM_AC/augmentation_examples',
                        help='Output directory')
    parser.add_argument('--config', type=str, 
                        default=os.path.join(PROJECT_ROOT, 'trainings', 'daft_exprt_tiny_aug', 'config.json'),
                        help='Config file for HParams')
    parser.add_argument('--vocoder_checkpoint', type=str, default='',
                        help='Path to HiFi-GAN checkpoint (optional)')
    
    args = parser.parse_args()
    
    # Load HParams
    if os.path.exists(args.config):
        with open(args.config) as f:
            import json
            data = f.read()
            config = json.loads(data)
        hparams = HyperParams(verbose=False, **config)
    else:
        print(f"Config file not found at {args.config}, using defaults")
        hparams = HyperParams(verbose=False)

    # Setup Output Dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Vocoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Vocoder on {device}...")
    vocoder = load_hifigan_vocoder(args.vocoder_checkpoint or None, device)
    
    # Load Audio
    print(f"Loading Audio: {args.input_wav}")
    wav, fs = librosa.load(args.input_wav, sr=hparams.sampling_rate)
    wav = rescale_wav_to_float32(wav)
    
    # Extract Mel
    mel_numpy = mel_spectrogram_HiFi(wav, hparams) # (80, T)
    mel_tensor = torch.from_numpy(mel_numpy).float().to(device).unsqueeze(0) # (1, 80, T)
    
    # Define Augmentations
    augmentations = [
        ('original', lambda m: m),
        ('pitch_up_3bins', lambda m: pitch_shift_bin_first(m, 3, hparams.min_clipping)),
        ('pitch_down_3bins', lambda m: pitch_shift_bin_first(m, -3, hparams.min_clipping)),
        ('fast_0.8x', lambda m: time_stretch(m, 0.8)),
        ('slow_1.2x', lambda m: time_stretch(m, 1.2)),
        ('quiet_0.7x', lambda m: energy_scale(m, 0.7)),
        ('loud_1.3x', lambda m: energy_scale(m, 1.3)),
        ('combo_fast_pitch_up', lambda m: time_stretch(pitch_shift_bin_first(m, 3, hparams.min_clipping), 0.8))
    ]
    
    print("Generating Augmentations...")
    for name, func in augmentations:
        print(f"Processing {name}...")
        
        # Apply Augmentation
        aug_mel = func(mel_tensor)
        
        # Vocode
        with torch.no_grad():
            # vocoder.infer expects (B, 80, T) or (80, T)? 
            # generate.py passes (80, T) from numpy but vocoder.infer likely handles it.
            # Let's pass (1, 80, T) to be safe as it's a batch of 1.
            # If vocoder expects (80, T), we squeeze.
            # Checking generate.py: it passes `mel_for_vocoder` which is `mel_spec` (numpy (80, T)).
            # So vocoder.infer likely handles numpy or tensor.
            # Let's pass tensor (1, 80, T).
            
            # Note: load_hifigan_vocoder returns a generator.
            # Usually HiFiGAN generator forward takes (B, 80, T).
            # But generate.py calls `vocoder.infer`.
            # If `vocoder` is the raw model, it's `vocoder(mel)`.
            # If it's a wrapper, it's `vocoder.infer(mel)`.
            # generate.py imports `load_hifigan_vocoder`.
            # Let's assume `vocoder` object has `infer` method if it's a wrapper, or `forward`.
            # generate.py says: `audio = vocoder.infer(mel_for_vocoder)`
            
            # Wait, if `load_hifigan_vocoder` returns the raw model, it doesn't have `infer`.
            # I should check `daft_exprt/vocoder/__init__.py`.
            # But I can't easily.
            # I'll try `vocoder(aug_mel)` if `infer` fails, or check if `vocoder` has `infer`.
            
            try:
                if hasattr(vocoder, 'infer'):
                     # generate.py passes numpy (80, T).
                     # If I pass tensor (1, 80, T), it might fail if it expects numpy.
                     # Let's try passing tensor first.
                    wav_out = vocoder.infer(aug_mel)
                else:
                    # Raw model usually takes (B, 80, T)
                    wav_out = vocoder(aug_mel)
                    if isinstance(wav_out, list) or isinstance(wav_out, tuple):
                        wav_out = wav_out[0]
                    wav_out = wav_out.squeeze()
            except Exception as e:
                print(f"Vocoder inference failed with tensor: {e}")
                print("Trying with numpy...")
                aug_mel_np = aug_mel.squeeze().cpu().numpy()
                wav_out = vocoder.infer(aug_mel_np)

        # Post-process
        if isinstance(wav_out, torch.Tensor):
            wav_out = wav_out.squeeze().cpu().numpy()
            
        # Save
        out_path = os.path.join(args.output_dir, f'{name}.wav')
        sf.write(out_path, wav_out, hparams.sampling_rate)
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
