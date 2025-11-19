#!/usr/bin/env python3
"""
Extract symbol-level duration, pitch, and energy curves from raw audio files.

The script:
    1. Reads a manifest describing (audio_path|transcript[|phoneme_sequence]) pairs.
    2. Reuses an MFA dictionary and runs G2P only for words missing from it.
    3. Runs Montreal Forced Aligner to obtain phoneme-level timings.
    4. Computes mel-spectrograms, energy, and REAPER-based pitch tracks.
    5. Aggregates features per symbol to mimic Daft-Exprt's local prosody predictor targets.
    6. Saves one line per audio with a list of tuples: (symbol, duration_frames, pitch, energy).

Requirements:
    - `mfa` command line interface installed and accessible on PATH.
    - MFA acoustic / G2P models downloaded (see `mfa download g2p ...` / `mfa download acoustic ...`).
    - `reaper` binary installed for pitch extraction.
"""
import argparse
import ast
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Set

import librosa
import numpy as np

FILE_ROOT = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_ROOT)
os.environ.setdefault('PYTHONPATH', os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from daft_exprt.cleaners import text_cleaner  # noqa: E402
from daft_exprt.extract_features import (  # noqa: E402
    duration_to_integer,
    extract_energy,
    extract_pitch,
    get_min_phone_duration,
    get_symbols_energy,
    get_symbols_pitch,
    mel_spectrogram_HiFi,
    rescale_wav_to_float32,
    update_markers,
)
from daft_exprt.hparams import HyperParams  # noqa: E402
from daft_exprt.mfa import extract_markers, prepare_corpus  # noqa: E402
from daft_exprt.symbols import eos, punctuation, whitespace  # noqa: E402


_logger = logging.getLogger(__name__)

# Mapping from logical presets to MFA g2p package names.
G2P_PRESETS = {
    'american_english': 'english_g2p',
    'indian_english': 'indic_english_g2p',
}

BOUNDARY_SYMBOLS = set(punctuation) | {whitespace, eos}

@dataclass
class AudioSample:
    """Container describing one audio/text pair."""

    audio_path: str
    transcript: str
    clean_transcript: str
    file_id: str
    custom_symbols: Sequence[str] = None


@dataclass
class SymbolProsody:
    """Symbol-level prosody statistics."""

    symbol: str
    duration: int
    pitch: float
    energy: float


def str2bool(value) -> bool:
    """argparse-compatible bool parser."""
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if lowered in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Cannot interpret "{value}" as boolean.')


def ensure_binary_available(binary: str) -> None:
    """Validate that an executable is available on PATH."""
    if shutil.which(binary) is None:
        raise RuntimeError(
            f'Required binary "{binary}" was not found on PATH. '
            f'Please install it or update PATH before running this script.'
        )


def parse_manifest(manifest_path: str, language: str, separator: str) -> List[AudioSample]:
    """Parse manifest lines formatted as "<audio_path>|<transcript>"."""
    samples: List[AudioSample] = []
    manifest_path = os.path.abspath(manifest_path)
    assert os.path.isfile(manifest_path), f'Manifest file "{manifest_path}" does not exist.'
    with open(manifest_path, 'r', encoding='utf-8') as manifest:
        for idx, raw_line in enumerate(manifest):
            line = raw_line.strip()
            # Skipping comments and empty lines
            if not line or line.startswith('#'):
                continue
            if separator not in line:
                raise ValueError(
                    f'Line {idx + 1} must contain separator "{separator}". Got: "{line}"'
                )
            parts = line.split(separator)
            if len(parts) < 2:
                raise ValueError(
                    f'Line {idx + 1}: expected at least "<audio>|<text>". Got: "{line}"'
                )
            audio_part = parts[0]
            text_part = parts[1]
            phoneme_part = separator.join(parts[2:]).strip() if len(parts) > 2 else ''
            audio_path = os.path.abspath(audio_part.strip())
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(f'Audio file "{audio_path}" referenced on line {idx + 1} is missing.')
            clean_sentence = text_cleaner(text_part.strip(), language).strip()
            if not clean_sentence:
                raise ValueError(f'Line {idx + 1} has an empty transcript after cleaning.')
            base_name = Path(audio_path).stem.replace(' ', '_')
            sample_id = f'{idx:04d}_{base_name}'
            custom_symbols = None
            if phoneme_part:
                custom_symbols = _parse_symbol_sequence(phoneme_part)
            samples.append(
                AudioSample(
                    audio_path=audio_path,
                    transcript=text_part.strip(),
                    clean_transcript=clean_sentence,
                    file_id=sample_id,
                    custom_symbols=custom_symbols or None,
                )
            )
    if not samples:
        raise RuntimeError(f'No valid samples found in manifest "{manifest_path}".')
    return samples


def _parse_symbol_sequence(payload: str) -> List[str]:
    """Interpret a user-provided phoneme sequence."""
    tokens = None
    try:
        parsed = ast.literal_eval(payload)
        if isinstance(parsed, (list, tuple)):
            tokens = [str(tok) for tok in parsed]
    except (SyntaxError, ValueError):
        tokens = None
    if tokens is None:
        tokens = [tok for tok in payload.strip().split() if tok]
    return tokens


def _apply_custom_symbols(sample: AudioSample, markers: List[List[str]]) -> None:
    """Override marker symbols using user-provided sequence."""
    custom_symbols = list(sample.custom_symbols or [])
    if not custom_symbols:
        return
    phoneme_indices = [idx for idx, marker in enumerate(markers) if marker[3] not in BOUNDARY_SYMBOLS]
    if len(custom_symbols) != len(phoneme_indices):
        raise RuntimeError(
            f'Custom phoneme count mismatch for "{sample.audio_path}". '
            f'Alignment contains {len(phoneme_indices)} non-boundary phonemes but {len(custom_symbols)} '
            f'symbols were provided. '
            'Ensure custom phoneme sequences match the MFA alignment length.'
        )
    for token, idx in zip(custom_symbols, phoneme_indices):
        markers[idx][3] = token
        markers[idx][4] = token


def resolve_g2p_model(args) -> str:
    """Return path to the desired G2P model, downloading instructions if missing."""
    if args.g2p_model:
        g2p_path = os.path.abspath(args.g2p_model)
    else:
        preset_name = G2P_PRESETS.get(args.g2p_preset, args.g2p_preset)
        home = str(Path.home())
        g2p_path = os.path.join(home, 'Documents', 'MFA', 'pretrained_models', 'g2p', f'{preset_name}.zip')
    if not os.path.isfile(g2p_path):
        raise FileNotFoundError(
            f'G2P model "{g2p_path}" was not found. '
            f'Use "mfa download g2p {os.path.basename(g2p_path).replace(".zip", "")}" to install it, '
            f'or pass --g2p_model with an explicit path.'
        )
    return g2p_path


def collect_manifest_vocabulary(samples: Sequence[AudioSample]) -> Set[str]:
    """Return lowercase vocabulary extracted from cleaned transcripts."""
    vocab: Set[str] = set()
    for sample in samples:
        tokens = re.findall(r"[a-z']+", sample.clean_transcript.lower())
        vocab.update([token for token in tokens if token])
    return vocab


def load_dictionary_words(dictionary_path: str) -> Set[str]:
    """Return the set of lowercase words already covered by an MFA dictionary."""
    words: Set[str] = set()
    with open(dictionary_path, 'r', encoding='utf-8') as dict_file:
        for line in dict_file:
            stripped = line.strip()
            if not stripped or stripped.startswith(';'):
                continue
            word = stripped.split()[0]
            words.add(word.lower())
    return words


def prepare_pronunciation_dictionary(
    samples: Sequence[AudioSample],
    base_dictionary: str,
    args,
    work_root: str,
) -> str:
    """Copy an existing dictionary and augment it with G2P pronunciations for OOV words."""
    base_dictionary = os.path.abspath(base_dictionary)
    assert os.path.isfile(base_dictionary), f'Pronunciation dictionary "{base_dictionary}" does not exist.'
    os.makedirs(work_root, exist_ok=True)
    combined_dictionary = os.path.join(work_root, 'combined.dict')
    shutil.copyfile(base_dictionary, combined_dictionary)

    vocab = collect_manifest_vocabulary(samples)
    covered_words = load_dictionary_words(base_dictionary)
    oov_words = sorted(vocab - covered_words)
    if len(oov_words) == 0:
        _logger.info('All manifest words already covered by dictionary.')
        return combined_dictionary

    g2p_model = resolve_g2p_model(args)
    _logger.info('Generating pronunciations for %d OOV words using MFA G2P...', len(oov_words))
    oov_list = os.path.join(work_root, 'oov_words.txt')
    with open(oov_list, 'w', encoding='utf-8') as vocab_file:
        for word in oov_words:
            vocab_file.write(f'{word}\n')
    oov_trans = os.path.join(work_root, 'oov_words_transcriptions.txt')
    tmp_dir = os.path.join(work_root, 'g2p_tmp')
    cmd = [
        'mfa',
        'g2p',
        g2p_model,
        oov_list,
        oov_trans,
        '-t',
        tmp_dir,
    ]
    subprocess.check_call(cmd)
    with open(oov_trans, 'r', encoding='utf-8') as g2p_file, \
            open(combined_dictionary, 'a', encoding='utf-8') as dict_file:
        for line in g2p_file:
            dict_file.write(line if line.endswith('\n') else f'{line}\n')
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.remove(oov_list)
    os.remove(oov_trans)

    return combined_dictionary


def prepare_corpus_dir(samples: Sequence[AudioSample], corpus_root: str) -> str:
    """Copy audio files and create metadata.csv expected by MFA."""
    wavs_dir = os.path.join(corpus_root, 'wavs')
    os.makedirs(wavs_dir, exist_ok=True)
    metadata_path = os.path.join(corpus_root, 'metadata.csv')
    with open(metadata_path, 'w', encoding='utf-8') as meta_file:
        for sample in samples:
            dest = os.path.join(wavs_dir, f'{sample.file_id}.wav')
            if not sample.audio_path.lower().endswith('.wav'):
                raise ValueError(f'Audio file "{sample.audio_path}" must be a .wav file for MFA alignment.')
            shutil.copy2(sample.audio_path, dest)
            meta_file.write(f'{sample.file_id}|{sample.transcript}\n')
    return metadata_path


def run_forced_alignment(
    corpus_dir: str,
    dictionary_path: str,
    acoustic_model: str,
    n_jobs: int,
    language: str,
) -> str:
    """Run MFA alignment and return directory containing .markers/.lab files."""
    _logger.info('Preparing MFA corpus...')
    prepare_corpus(corpus_dir, language)
    align_dir = os.path.join(corpus_dir, 'align')
    tmp_dir = os.path.join(corpus_dir, 'tmp_align')
    os.makedirs(tmp_dir, exist_ok=True)
    cmd = [
        'mfa',
        'align',
        corpus_dir,
        dictionary_path,
        acoustic_model,
        align_dir,
        '-t',
        tmp_dir,
        '-j',
        str(n_jobs),
        '-v',
        '-c',
    ]
    _logger.info('Running MFA forced alignment...')
    subprocess.check_call(cmd)
    _logger.info('Extracting phone/word markers from TextGrid files...')
    text_grid_dir = os.path.join(align_dir, 'wavs')
    extract_markers(text_grid_dir, max(n_jobs, 1))
    lab_src_dir = os.path.join(corpus_dir, 'wavs')
    for file_name in os.listdir(lab_src_dir):
        if file_name.endswith('.lab'):
            shutil.move(os.path.join(lab_src_dir, file_name), os.path.join(text_grid_dir, file_name))
    return text_grid_dir


def compute_symbol_prosody(
    sample: AudioSample,
    markers_dir: str,
    corpus_dir: str,
    hparams: HyperParams,
) -> List[SymbolProsody]:
    """Compute per-symbol prosody values for a single sample."""
    markers_path = os.path.join(markers_dir, f'{sample.file_id}.markers')
    assert os.path.isfile(markers_path), f'Markers file "{markers_path}" not found.'
    with open(markers_path, 'r', encoding='utf-8') as marker_file:
        lines = [line.strip().split('\t') for line in marker_file if line.strip()]
    min_phone_dur = get_min_phone_duration(['\t'.join(line) for line in lines])
    fft_length = hparams.filter_length / hparams.sampling_rate
    if not min_phone_dur > fft_length / 2:
        raise RuntimeError(
            f'{markers_path} contains phones shorter than FFT/2 ({min_phone_dur:.4f}s vs {fft_length / 2:.4f}s).'
        )

    sent_begin = float(lines[0][0])
    sent_end = float(lines[-1][1])
    wav_path = os.path.join(corpus_dir, 'wavs', f'{sample.file_id}.wav')
    wav, _ = librosa.load(wav_path, sr=hparams.sampling_rate)
    wav = rescale_wav_to_float32(wav)
    wav = wav[int(sent_begin * hparams.sampling_rate): int(sent_end * hparams.sampling_rate)]
    mel_spec = mel_spectrogram_HiFi(wav, hparams)
    nb_mel_frames = mel_spec.shape[1]
    float_durations = [[float(line[0]) - sent_begin, float(line[1]) - sent_begin] for line in lines]
    int_durations = duration_to_integer(float_durations, hparams, nb_samples=len(wav))
    if sum(int_durations) != nb_mel_frames:
        raise RuntimeError(
            f'{markers_path} duration/frame mismatch ({sum(int_durations)} vs {nb_mel_frames}).'
        )
    lab_path = os.path.join(markers_dir, f'{sample.file_id}.lab')
    with open(lab_path, 'r', encoding='utf-8') as lab_file:
        sentence = lab_file.readline()
    markers = update_markers(sample.file_id, ['\t'.join(line) for line in lines],
                             sentence, sent_begin, int_durations, hparams, _logger)
    if markers is None:
        raise RuntimeError(f'Unable to align sentence/markers for "{sample.file_id}".')
    if sample.custom_symbols:
        _apply_custom_symbols(sample, markers)
    mel_linear = np.exp(mel_spec)
    energy_per_frame = extract_energy(mel_linear)
    pitch_per_frame = extract_pitch(wav, hparams.sampling_rate, hparams)
    if len(energy_per_frame) != nb_mel_frames or len(pitch_per_frame) != nb_mel_frames:
        raise RuntimeError(f'{sample.file_id}: mismatch between mel frames and energy/pitch frames.')
    symbol_energy = [float(val.strip()) for val in get_symbols_energy(energy_per_frame, markers)]
    symbol_pitch = [float(val.strip()) for val in get_symbols_pitch(pitch_per_frame, markers)]
    prosody = []
    for marker, pitch_val, energy_val in zip(markers, symbol_pitch, symbol_energy):
        prosody.append(
            SymbolProsody(
                symbol=marker[3],
                duration=int(marker[2]),
                pitch=round(pitch_val, 3),
                energy=round(energy_val, 3),
            )
        )
    return prosody


def format_output_line(
    sample: AudioSample,
    prosody: Sequence[SymbolProsody],
    include_path: bool,
    mix_output: bool,
) -> str:
    """Serialize prosody tuples for one sample."""
    if mix_output:
        tuples_str = ', '.join(
            f"('{p.symbol}', {p.duration}, {p.pitch:.3f}, {p.energy:.3f})" for p in prosody
        )
        payload_repr = f'[{tuples_str}]'
    else:
        symbols = [p.symbol for p in prosody]
        durations = [p.duration for p in prosody]
        pitch_vals = [round(p.pitch, 3) for p in prosody]
        energy_vals = [round(p.energy, 3) for p in prosody]
        payload_repr = repr((symbols, durations, pitch_vals, energy_vals))
    if include_path:
        return f'{sample.audio_path}|{payload_repr}'
    return payload_repr


def build_hparams(language: str, scratch_dir: str) -> HyperParams:
    """Create a HyperParams instance with minimal required fields."""
    os.makedirs(scratch_dir, exist_ok=True)
    return HyperParams(
        verbose=False,
        training_files='',
        validation_files='',
        output_directory=scratch_dir,
        language=language,
        speakers=['inference'],
        speakers_id=[0],
    )


def main():
    parser = argparse.ArgumentParser(description='Extract symbol-level prosody statistics from audio files.')
    parser.add_argument('--manifest', required=True,
                        help='Text file with one "<audio_path>|<transcript>" pair per line.')
    parser.add_argument('--output', required=True,
                        help='Destination txt file. Each line contains tuples for one audio.')
    parser.add_argument('--separator', default='|',
                        help='Separator used between audio path and transcript in the manifest.')
    parser.add_argument('--language', default='english', help='Language used by MFA/text cleaner.')
    parser.add_argument('--g2p_preset', default='american_english',
                        help='Logical G2P preset (american_english or indian_english).')
    parser.add_argument('--g2p_model', default='',
                        help='Explicit path to an MFA G2P model. Overrides --g2p_preset when set.')
    parser.add_argument('--dictionary', default='',
                        help='Path to an MFA pronunciation dictionary. Defaults to the pretrained config dictionary.')
    parser.add_argument('--acoustic_model', default='',
                        help='Path to an MFA acoustic model zip. Defaults to MFA English model.')
    parser.add_argument('--nb_jobs', default=4, type=int, help='Number of parallel jobs for MFA.')
    parser.add_argument('--work_dir', default=os.path.join(PROJECT_ROOT, 'tmp_symbol_prosody'),
                        help='Scratch directory for intermediate files.')
    parser.add_argument('--keep_temps', action='store_true', help='Keep intermediate files for debugging.')
    parser.add_argument('--include_audio_path', action='store_true',
                        help='Prefix each output line with the original audio path.')
    parser.add_argument('--mix_output', type=str2bool, default=True,
                        help='When true (default), emit tuples per symbol; when false, emit separate arrays '
                             'for symbols, duration, pitch, and energy.')
    args = parser.parse_args()

    logging.basicConfig(
        handlers=[logging.StreamHandler()],
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    ensure_binary_available('mfa')
    ensure_binary_available('reaper')

    samples = parse_manifest(args.manifest, args.language, args.separator)
    scratch_root = os.path.join(args.work_dir, f'extract_{uuid.uuid4().hex}')
    corpus_dir = os.path.join(scratch_root, 'corpus')
    os.makedirs(corpus_dir, exist_ok=True)

    try:
        hparams = build_hparams(args.language, scratch_root)
        base_dictionary = os.path.abspath(args.dictionary) if args.dictionary else hparams.mfa_dictionary
        dictionary_path = prepare_pronunciation_dictionary(samples, base_dictionary, args, scratch_root)
        if args.acoustic_model:
            acoustic_model_path = os.path.abspath(args.acoustic_model)
        else:
            acoustic_model_path = hparams.mfa_acoustic_model
        if not os.path.isfile(acoustic_model_path):
            raise FileNotFoundError(
                f'Acoustic model "{acoustic_model_path}" was not found. '
                f'Use "mfa download acoustic {os.path.basename(acoustic_model_path).replace(".zip", "")}" '
                f'or pass --acoustic_model.'
            )
        prepare_corpus_dir(samples, corpus_dir)
        markers_dir = run_forced_alignment(corpus_dir, dictionary_path, acoustic_model_path, args.nb_jobs, args.language)

        results = []
        for sample in samples:
            _logger.info('Processing "%s"...', sample.file_id)
            prosody = compute_symbol_prosody(sample, markers_dir, corpus_dir, hparams)
            results.append((sample, prosody))

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as out_file:
            for sample, prosody in results:
                out_file.write(format_output_line(sample, prosody, args.include_audio_path, args.mix_output) + '\n')
        _logger.info('Done. Results saved to "%s".', args.output)
    finally:
        if args.keep_temps:
            _logger.info('Keeping temporary directory at %s', scratch_root)
        else:
            shutil.rmtree(scratch_root, ignore_errors=True)


if __name__ == '__main__':
    main()
