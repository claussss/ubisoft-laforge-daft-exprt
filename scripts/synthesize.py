import argparse
import ast
import json
import logging
import os
import random
import sys
import time

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from shutil import rmtree

FILE_ROOT = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_ROOT)
os.environ['PYTHONPATH'] = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from daft_exprt.generate import extract_reference_parameters, generate_mel_specs, prepare_sentences_for_inference
from daft_exprt.extract_features import (
    extract_energy as compute_energy_frames,
    extract_pitch,
    extract_energy, # Added for direct use in new_speaker_stats logic
    mel_spectrogram_HiFi,
    rescale_wav_to_float32,
)
from daft_exprt.hparams import HyperParams
from daft_exprt.model import DaftExprt
from daft_exprt.utils import get_nb_jobs


_logger = logging.getLogger(__name__)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1234)


'''
    Script example that showcases how to generate with Daft-Exprt
    using a target sentence, a target speaker, and a target prosody
'''


def _looks_like_split_arrays(payload):
    if not (isinstance(payload, (list, tuple)) and len(payload) == 4):
        return False
    lengths = []
    for arr in payload:
        if not isinstance(arr, (list, tuple)):
            return False
        lengths.append(len(arr))
    return len(set(lengths)) == 1


def _coerce_split_arrays_to_tuples(payload, line_idx):
    if not _looks_like_split_arrays(payload):
        return None
    symbols, durations, pitch, energy = payload
    if len(symbols) == 0:
        raise ValueError(f'Line {line_idx}: Empty symbol list.')
    if not (len(symbols) == len(durations) == len(pitch) == len(energy)):
        raise ValueError(f'Line {line_idx}: Arrays must share the same length.')
    return list(zip(symbols, durations, pitch, energy))


def parse_symbol_prosody_file(prosody_file, hparams):
    '''Load symbol-level prosody tuples saved by extract_symbol_prosody.py'''
    assert os.path.isfile(prosody_file), f'There is no such file {prosody_file}'
    sentences, file_names, external_prosody = [], [], []
    with open(prosody_file, 'r', encoding='utf-8') as f:
        for line_idx, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            source_hint, payload = None, line
            if '|' in line:
                maybe_source, remainder = line.split('|', 1)
                trimmed = remainder.lstrip()
                if trimmed.startswith('[') or trimmed.startswith('('):
                    source_hint = maybe_source.strip()
                    payload = trimmed
            try:
                parsed_payload = ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                raise ValueError(f'Line {line_idx}: Unable to parse symbol prosody tuples.')
            tuples = None
            if isinstance(parsed_payload, (list, tuple)):
                if parsed_payload and all(
                    isinstance(entry, (list, tuple)) and len(entry) == 4 for entry in parsed_payload
                ):
                    tuples = parsed_payload
                else:
                    tuples = _coerce_split_arrays_to_tuples(parsed_payload, line_idx)
            if tuples is None:
                raise ValueError(f'Line {line_idx}: Expected a list of tuples.')
            symbols, durations, pitch, energy = [], [], [], []
            for entry in tuples:
                if not (isinstance(entry, (list, tuple)) and len(entry) == 4):
                    raise ValueError(f'Line {line_idx}: Each tuple must be (symbol, duration, pitch, energy).')
                symbol, dur, pitch_val, energy_val = entry
                symbol = str(symbol)
                if symbol not in hparams.symbols:
                    raise ValueError(f'Line {line_idx}: Symbol "{symbol}" is not part of the configured symbol set.')
                symbols.append(symbol)
                durations.append(int(dur))
                pitch.append(float(pitch_val))
                energy.append(float(energy_val))
            if len(symbols) == 0:
                raise ValueError(f'Line {line_idx}: Empty symbol list.')
            if source_hint:
                base_name = os.path.splitext(os.path.basename(source_hint))[0]
            else:
                base_name = f'symbol_prosody_line{len(file_names)}'
            sentences.append(list(symbols))
            file_names.append(base_name)
            external_prosody.append({
                'symbols': list(symbols),
                'durations_frames': durations,
                'pitch': pitch,
                'energy': energy
            })
    if len(sentences) == 0:
        raise ValueError(f'File "{prosody_file}" does not contain any symbol prosody entries.')
    return sentences, file_names, external_prosody


def prepare_symbol_prosody_inputs(prosody_file, output_dir, hparams):
    '''Reset output directory and load symbol-level prosody inputs'''
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False) # External prosody is the one from symbol_prosody_file
    sentences, file_names, external_prosody = parse_symbol_prosody_file(prosody_file, hparams)
    with open(os.path.join(output_dir, 'sentences_to_generate.txt'), 'w', encoding='utf-8') as f:
        for file_name, tokens in zip(file_names, sentences):
            text = ' '.join(tokens)
            f.write(f'{file_name}|{text}\n')
    return sentences, file_names, external_prosody


def _load_manifest(path, expected_len, split_text=False):
    refs = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if split_text and '|' in line:
                line = line.split('|', 1)[0]
            refs.append(os.path.abspath(line))
    if expected_len is not None and len(refs) != expected_len:
        raise ValueError(f'Manifest "{path}" has {len(refs)} entries but expected {expected_len}.')
    for ref in refs:
        if not os.path.isfile(ref):
            raise FileNotFoundError(f'Audio "{ref}" listed in {path} does not exist.')
    return refs


def _sanitize_ref_path(path):
    abs_path = os.path.abspath(path)
    marker = os.sep + 'datasets' + os.sep
    if marker in abs_path:
        abs_path = abs_path.split(marker, 1)[1]
    for ch in ['/', '\\', ':']:
        abs_path = abs_path.replace(ch, '_')
    return abs_path


def _load_speaker_stats(stats_path):
    with open(stats_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    for key in ('pitch', 'energy'):
        if key not in stats or 'mean' not in stats[key] or 'std' not in stats[key]:
            raise ValueError(f'Stats file "{stats_path}" is missing "{key}.mean"/"{key}.std".')
        if stats[key]['std'] == 0:
            raise ValueError(f'Stats file "{stats_path}" has zero std for "{key}".')
    return stats


def synthesize(args, dur_factor=None, energy_factor=None, pitch_factor=None, 
               pitch_transform=None, use_griffin_lim=False, get_time_perf=False):
    ''' Generate with DaftExprt
    '''
    # get hyper-parameters that were used to create the checkpoint
    checkpoint_dict = torch.load(args.checkpoint, map_location=f'cuda:{0}')
    hparams = HyperParams(verbose=False, **checkpoint_dict['config_params'])
    # load model
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    model = DaftExprt(hparams).to(device)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)

    # prepare sentences
    n_jobs = 1 #get_nb_jobs('max')
    if args.symbol_prosody_file:
        sentences, file_names, external_prosody = prepare_symbol_prosody_inputs(args.symbol_prosody_file, args.output_dir, hparams)
    else:
        sentences, file_names = prepare_sentences_for_inference(args.text_file, args.output_dir, hparams, n_jobs)
        external_prosody = None

    # update hparams with new speaker stats
    external_embeddings = None
    if args.new_speaker_stats is not None:
        if os.path.isdir(args.new_speaker_stats):
             _logger.info(f"Computing stats from directory: {args.new_speaker_stats}")
             # Directory mode: Scan for wavs, compute stats and avg embedding (ECAPA)
             try:
                 from speechbrain.inference.speaker import EncoderClassifier
             except ImportError:
                 from speechbrain.pretrained import EncoderClassifier
             
             # Load Classifier
             try:
                 classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
             except Exception as e:
                 _logger.error(f"Failed to load ECAPA model: {e}")
                 raise e

             wav_files = [os.path.join(args.new_speaker_stats, f) for f in os.listdir(args.new_speaker_stats) if f.endswith('.wav')]
             if not wav_files:
                 raise ValueError(f"No .wav files found in {args.new_speaker_stats}")
                 
             # Compute Stats and Embeddings
             all_embs = []
             
             all_pitch = []
             all_energy = []
             
             for wav_path in wav_files:
                 # Load Audio
                 signal, fs = torchaudio.load(wav_path)
                 
                 # Energy/Pitch
                 # extract_features functions take (wav, fs, hparams)
                 # wav should be float numpy
                 wav_np = signal.squeeze().numpy()
                 
                 try:
                     p = extract_pitch(wav_np, fs, hparams)
                     e = extract_energy(wav_np, fs, hparams)
                     
                     all_pitch.extend(p[p > 0])
                     all_energy.extend(e[e > 0])
                 except Exception as err:
                     _logger.warning(f"Feature extraction failed for {wav_path}: {err}")
                 
                 # Embedding
                 # Resample for ECAPA if needed (16k)
                 if fs != 16000:
                     resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                     signal_16k = resampler(signal)
                 else:
                     signal_16k = signal
                     
                 emb = classifier.encode_batch(signal_16k.to(device))
                 all_embs.append(emb.squeeze().cpu().numpy())
                 
             # Avg Stats
             if len(all_pitch) > 0:
                 p_mean = float(np.mean(all_pitch))
                 p_std = float(np.std(all_pitch))
             else:
                 p_mean, p_std = 0.0, 1.0 # fallback
                 
             if len(all_energy) > 0:
                 e_mean = float(np.mean(all_energy))
                 e_std = float(np.std(all_energy))
             else:
                 e_mean, e_std = 0.0, 1.0
            
             # Avg Embedding
             avg_emb = np.mean(np.array(all_embs), axis=0) # (192,)
             external_embeddings = torch.from_numpy(avg_emb).unsqueeze(0).to(device) # (1, 192)
             
             # Format into hparams.stats structure
             if not hasattr(hparams, 'stats'):
                 hparams.stats = {}
             hparams.stats['spk 0'] = {
                 'pitch': {'mean': p_mean, 'std': p_std},
                 'energy': {'mean': e_mean, 'std': e_std}
             }
             
        elif os.path.isfile(args.new_speaker_stats):
             stats = _load_speaker_stats(args.new_speaker_stats)
             if not hasattr(hparams, 'stats'):
                 hparams.stats = {}
             # Assuming a single speaker stats file, apply to a dummy ID '0'
             hparams.stats['spk 0'] = stats
             _logger.info(f"Loaded speaker stats from file: {args.new_speaker_stats}")
    
    # Enable external embeddings flag for inference if we have them
    if external_embeddings is not None:
        hparams.use_external_embeddings = True

    source_stats = hparams.stats.get('spk 0') if args.new_speaker_stats else None # This is for normalizing symbol_prosody input
    refs = []
    if args.symbol_prosody_file and os.path.isfile(args.style_bank):
        manifest_path = os.path.abspath(args.style_bank)
        ref_dir = os.path.dirname(manifest_path)
        manifest_refs = _load_manifest(manifest_path, len(sentences))
        for audio_ref in manifest_refs:
            ref_name = _sanitize_ref_path(audio_ref)
            extract_reference_parameters(audio_ref, ref_dir or '.', hparams, ref_name=ref_name)
            refs.append(os.path.join(ref_dir or '.', f'{ref_name}.npz'))
    else:
        # extract reference parameters from directory of wavs
        audio_refs = [os.path.join(args.style_bank, x) for x in os.listdir(args.style_bank) if x.endswith('.wav')]
        for audio_ref in audio_refs:
            extract_reference_parameters(audio_ref, args.style_bank, hparams)
        cache_refs = [os.path.join(args.style_bank, x) for x in os.listdir(args.style_bank) if x.endswith('.npz')]
        refs = [random.choice(cache_refs) for _ in range(len(sentences))]
    
    # Use speaker_id 0 to match our injected stats if in zero-shot mode
    if args.new_speaker_stats is not None and (os.path.isdir(args.new_speaker_stats) or os.path.isfile(args.new_speaker_stats)):
        speaker_ids = [0 for _ in range(len(sentences))]
    else:
        speaker_ids = [12 for _ in range(len(sentences))] # Fallback/Legacy
    
    vocoder = None
    if not use_griffin_lim:
        from daft_exprt.vocoder import load_hifigan_vocoder
        device = next(model.parameters()).device
        vocoder = load_hifigan_vocoder(args.vocoder_checkpoint or None, device)

    # add duration factors for each symbol in the sentence
    dur_factors = [] if dur_factor is not None else None
    energy_factors = [] if energy_factor is not None else None
    pitch_factors = [pitch_transform, []] if pitch_factor is not None else None
    for sentence in sentences:
        # count number of symbols in the sentence
        nb_symbols = 0
        for item in sentence:
            if isinstance(item, list):  # correspond to phonemes of a word
                nb_symbols += len(item)
            else:  # correspond to word boundaries
                nb_symbols += 1
        # append to lists
        if dur_factors is not None:
            dur_factors.append([dur_factor for _ in range(nb_symbols)])
        if energy_factors is not None:
            energy_factors.append([energy_factor for _ in range(nb_symbols)])
        if pitch_factors is not None:
            pitch_factors[1].append([pitch_factor for _ in range(nb_symbols)])

    # Expand external embeddings to batch size
    batch_ext_embs = None
    if external_embeddings is not None:
        batch_ext_embs = external_embeddings.repeat(len(sentences), 1)
        
    # generate mel-specs and synthesize audios with Griffin-Lim
    predictions = generate_mel_specs(model, sentences, file_names, speaker_ids, refs, args.output_dir,
                       hparams, dur_factors, energy_factors, pitch_factors, args.batch_size,
                       n_jobs, use_griffin_lim, get_time_perf, external_prosody, vocoder,
                       source_stats=source_stats, reduce_buzz=args.reduce_buzz,
                       neutralize_prosody=args.neutralize_prosody,
                       neutralize_speaker_encoder=args.neutralize_speaker_encoder,
                       alpha_dur=args.alpha_dur, alpha_pitch=args.alpha_pitch, alpha_energy=args.alpha_energy,
                       external_embeddings=batch_ext_embs)
    compare_paths = None
    if args.plot_prosody_files_to_compare:
        compare_paths = _load_manifest(os.path.abspath(args.plot_prosody_files_to_compare), len(sentences), split_text=True)
    if compare_paths and predictions is not None:
        final_names = []
        for base, speaker_id, ref in zip(file_names, speaker_ids, refs):
            ref_name = os.path.basename(ref).replace('.npz', '')
            final_names.append(f'{base}_spk_{speaker_id}_ref_{ref_name}')
        try:
            _plot_prosody_curves(sentences, final_names, refs, predictions,
                                 speaker_ids, args.output_dir, hparams,
                                 external_prosody, compare_paths)
        except Exception as exc:
            _logger.warning(f'Failed to plot prosody curves: {exc}')
    
    return file_names, refs, speaker_ids


def pair_ref_and_generated(args, file_names, refs, speaker_ids):
    ''' Simplify prosody transfer evaluation by matching generated audio with its reference
    '''
    # save references to output dir to make prosody transfer evaluation easier
    for idx, (file_name, ref, speaker_id) in enumerate(zip(file_names, refs, speaker_ids)):
        # extract reference id (for logging only; we no longer copy audio)
        ref_file_name = os.path.basename(ref).replace('.npz', '')

        # TODO: delete later
        # check correponding synthesized audio exists
        synthesized_file_name = f'{file_name}_spk_{speaker_id}_ref_{ref_file_name}'
        synthesized_audio = os.path.join(args.output_dir, f'{synthesized_file_name}.wav')
        assert(os.path.isfile(synthesized_audio)), _logger.error(f'There is no such file {synthesized_audio}')
        # rename files
        os.rename(synthesized_audio, f'{os.path.join(args.output_dir, f"{idx}_{synthesized_file_name}.wav")}')

def _plot_prosody_curves(sentences, final_names, refs, predictions, speaker_ids,
                         output_dir, hparams, external_prosody, compare_paths):
    """Plot frame-level and symbol-level prosody comparisons."""
    for idx, (file_name, ref_path) in enumerate(zip(final_names, refs)):
        if file_name not in predictions:
            continue
        mel_spec = predictions[file_name][4]
        energy_gen = compute_energy_frames(np.exp(mel_spec)) # Frame level predicted energy
        comp_audio = compare_paths[idx]
        energy_ref, pitch_ref = _compute_reference_features(ref_path, comp_audio, hparams) # Frame level reference
        wav_path = os.path.join(output_dir, f'{file_name}.wav')
        pitch_gen = None
        if os.path.isfile(wav_path):
            wav, _ = librosa.load(wav_path, sr=hparams.sampling_rate)
            wav = rescale_wav_to_float32(wav)
            pitch_gen = extract_pitch(wav, hparams.sampling_rate, hparams) # Frame level predicted pitch
        if pitch_gen is None or energy_ref is None or pitch_ref is None:
            continue
        durations_int = np.asarray(predictions[file_name][1])
        flat_symbols = _flatten_sentence_tokens(sentences[idx])
        if len(flat_symbols) == 0:
            continue
        symbol_durations = durations_int[:len(flat_symbols)]
        symbol_pitch_ref = _aggregate_symbol_values(pitch_ref, symbol_durations)
        symbol_pitch_gen = _aggregate_symbol_values(pitch_gen, symbol_durations)
        symbol_energy_ref = _aggregate_symbol_values(energy_ref, symbol_durations)
        symbol_energy_gen = _aggregate_symbol_values(energy_gen, symbol_durations)

        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        _plot_series(axes[0], energy_ref, energy_gen, 'Energy', f'Energy - {file_name}')
        _plot_series(axes[1], pitch_ref, pitch_gen, 'Log Pitch', f'Pitch - {file_name}')
        _plot_symbol_pitch(flat_symbols, axes[2], symbol_pitch_ref, symbol_pitch_gen, ylabel='Log Pitch',
                           title=f'Symbol Pitch - {file_name}')
        _plot_symbol_pitch(flat_symbols, axes[3], symbol_energy_ref, symbol_energy_gen, ylabel='Energy',
                           title=f'Symbol Energy - {file_name}')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{file_name}_prosody.png'))
        plt.close(fig)


def _compute_corr(ref_vals, gen_vals):
    ref_vals = np.asarray(ref_vals, dtype=float)
    gen_vals = np.asarray(gen_vals, dtype=float)
    mask = np.isfinite(ref_vals) & np.isfinite(gen_vals)
    if np.count_nonzero(mask) < 2:
        return None
    ref_vals = ref_vals[mask]
    gen_vals = gen_vals[mask]
    if np.std(ref_vals) == 0 or np.std(gen_vals) == 0:
        return None
    return float(np.corrcoef(ref_vals, gen_vals)[0, 1])


def _plot_series(axis, ref_vals, gen_vals, ylabel, title):
    ref_vals = np.asarray(ref_vals)
    gen_vals = np.asarray(gen_vals)
    length = min(len(ref_vals), len(gen_vals))
    if length == 0:
        return False
    ref_vals = ref_vals[:length]
    gen_vals = gen_vals[:length]
    ref_vals = np.where(ref_vals > 0, ref_vals, np.nan)
    gen_vals = np.where(gen_vals > 0, gen_vals, np.nan)
    corr = _compute_corr(ref_vals, gen_vals)
    axis.plot(range(length), ref_vals, label='Reference', linewidth=1, color='tab:blue')
    axis.plot(range(length), gen_vals, label='Generated', linewidth=1, color='tab:orange')
    axis.set_xlabel('Frame Index')
    axis.set_ylabel(ylabel)
    if corr is not None:
        axis.set_title(f'{title} (r={corr:.2f})')
        axis.text(0.02, 0.9, f'r={corr:.2f}', transform=axis.transAxes,
                  fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    else:
        axis.set_title(title)
    axis.legend()
    return True


def _flatten_sentence_tokens(sentence):
    tokens = []
    for item in sentence:
        if isinstance(item, list):
            tokens.extend(item)
        else:
            tokens.append(item)
    return tokens


def _plot_symbol_pitch(symbols, axis, ref_pitch, gen_pitch, ylabel='Log Pitch', title='Symbol Pitch'):
    filtered_symbols, ref_vals, gen_vals = [], [], []
    for sym, ref, gen in zip(symbols, ref_pitch, gen_pitch):
        if sym.strip() in {'', '~'}:
            continue
        filtered_symbols.append(sym)
        ref_vals.append(ref)
        gen_vals.append(gen)
    length = len(filtered_symbols)
    if length == 0:
        axis.set_visible(False)
        return
    ref_arr = np.asarray(ref_vals, dtype=float)
    gen_arr = np.asarray(gen_vals, dtype=float)
    corr = _compute_corr(ref_arr, gen_arr)
    axis.plot(range(length), ref_vals, label='Reference', marker='x', color='tab:blue')
    axis.plot(range(length), gen_vals, label='Generated', marker='o', color='tab:orange')
    axis.set_xticks(range(length))
    axis.set_xticklabels(filtered_symbols, rotation=90)
    axis.set_ylabel(ylabel)
    if corr is not None:
        axis.set_title(f'{title} (r={corr:.2f})')
        axis.text(0.02, 0.85, f'r={corr:.2f}', transform=axis.transAxes,
                  fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    else:
        axis.set_title(title)
    axis.legend()

def _aggregate_symbol_values(frame_values, durations_int):
    frame_values = np.asarray(frame_values)
    durations = np.asarray(durations_int, dtype=int)
    symbol_pitch = []
    idx = 0
    total = len(frame_values)
    for dur in durations:
        if dur <= 0:
            symbol_pitch.append(np.nan)
            continue
        end = min(idx + dur, total)
        frame_slice = frame_values[idx:end]
        idx = end
        voiced = frame_slice[frame_slice > 0]
        symbol_pitch.append(np.mean(voiced) if voiced.size else np.nan)
    return symbol_pitch


def _compute_reference_features(ref_npz_path, compare_audio, hparams):
    if compare_audio:
        wav, _ = librosa.load(compare_audio, sr=hparams.sampling_rate)
        wav = rescale_wav_to_float32(wav)
        mel = mel_spectrogram_HiFi(wav, hparams)
        energy = compute_energy_frames(np.exp(mel))
        pitch = extract_pitch(wav, hparams.sampling_rate, hparams)
        return energy, pitch
    ref_npz = np.load(ref_npz_path)
    return ref_npz['energy'], ref_npz['pitch']



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to synthesize sentences with Daft-Exprt')

    parser.add_argument('-out', '--output_dir', type=str,
                        help='output dir to store synthesis outputs')
    parser.add_argument('-chk', '--checkpoint', type=str,
                        help='checkpoint path to use for synthesis')
    parser.add_argument('-tf', '--text_file', type=str, default=os.path.join(PROJECT_ROOT, 'scripts', 'benchmarks', 'english', 'sentences.txt'),
                        help='text file to use for synthesis')
    parser.add_argument('-spf', '--symbol_prosody_file', type=str, default='',
                        help='optional file listing (symbol, duration, pitch, energy) tuples to bypass the prosody predictor')
    parser.add_argument('-sb', '--style_bank', type=str, default=os.path.join(PROJECT_ROOT, 'scripts', 'style_bank', 'english'),
                        help='directory path containing the reference utterances to use for synthesis')
    parser.add_argument('--use_griffin_lim', action='store_true',
                        help='use Griffin-Lim for waveform reconstruction instead of HiFi-GAN')
    parser.add_argument('--vocoder_checkpoint', type=str, default='',
                        help='optional path to a HiFi-GAN generator checkpoint (defaults to the universal model)')
    parser.add_argument('--reduce_buzz', action='store_true',
                        help='apply light mel smoothing and gentle low-pass before HiFi-GAN vocoding')
    parser.add_argument('-bs', '--batch_size', type=int, default=50,
                        help='batch of sentences to process in parallel')
    parser.add_argument('-rtf', '--real_time_factor', action='store_true',
                        help='get Daft-Exprt real time factor performance given the batch size')
    parser.add_argument('-ctrl', '--control', action='store_true',
                        help='perform local prosody control during synthesis')
    parser.add_argument('--new_speaker_stats', type=str, default='',
                        help='Optional path to JSON stats (mean/std) for an external speaker whose prosody was extracted.')
    parser.add_argument('--plot_prosody_files_to_compare', type=str, default='',
                        help='Path to a manifest (one audio path per line) whose files will be used for prosody comparison plots.')
    parser.add_argument('--neutralize_prosody', action='store_true',
                        help='Neutralize the reference audio embedding (zero it out).')
    parser.add_argument('--neutralize_speaker_encoder', action='store_true',
                        help='Neutralize the speaker embedding for the phoneme encoder only.')
    parser.add_argument('--alpha_dur', type=float, default=1.0,
                        help='Duration exaggeration factor (variance scaling). Default 1.0 (no change).')
    parser.add_argument('--alpha_pitch', type=float, default=1.0,
                        help='Pitch exaggeration factor (variance scaling). Default 1.0 (no change).')
    parser.add_argument('--alpha_energy', type=float, default=1.0,
                        help='Energy exaggeration factor (variance scaling). Default 1.0 (no change).')
    
    args = parser.parse_args()
    
    # set logger config
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
        ],
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    
    if args.real_time_factor:
        synthesize(args, get_time_perf=True, use_griffin_lim=args.use_griffin_lim)
        time.sleep(5)
        _logger.info('')
    if args.control:
        # small hard-coded example that showcases duration and pitch control
        # control is performed on the sentence level in this example
        # however, the code also supports control on the word/phoneme level
        dur_factor = 1.25  # decrease speed
        pitch_transform = 'add'  # pitch shift
        pitch_factor = 50  # 50Hz
        synthesize(args, dur_factor=dur_factor, pitch_factor=pitch_factor,
                   pitch_transform=pitch_transform, use_griffin_lim=args.use_griffin_lim)
    else:
        file_names, refs, speaker_ids = synthesize(args, use_griffin_lim=args.use_griffin_lim)
        pair_ref_and_generated(args, file_names, refs, speaker_ids)
