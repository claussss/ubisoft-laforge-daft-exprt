"""Compute ECAPA-TDNN speaker embeddings for preprocessed file lists. Used by training pre_process."""

import logging
import os

import numpy as np
import torch
import torchaudio

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier


_logger = logging.getLogger(__name__)


def compute_ecapa_for_file_lists(data_set_dir, training_files, validation_files, device=None):
    """Compute and save .spk_emb.npy for each line in training_files and validation_files.

    Line format: features_dir|feature_file|speaker_id
    Wav path: data_set_dir / speaker / wavs / {feature_file}.wav with speaker = os.path.basename(features_dir).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _logger.info("Loading SpeechBrain ECAPA-TDNN model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}
    )
    count = 0
    for file_list in [training_files, validation_files]:
        if not os.path.isfile(file_list):
            _logger.warning("File list %s not found, skipping.", file_list)
            continue
        with open(file_list, "r", encoding="utf-8") as f:
            lines = [line.strip().split("|") for line in f]
        for parts in lines:
            if len(parts) < 3:
                continue
            features_dir = parts[0]
            feature_file = parts[1]
            speaker = os.path.basename(os.path.normpath(features_dir))
            wav_path = os.path.join(data_set_dir, speaker, "wavs", f"{feature_file}.wav")
            if not os.path.isfile(wav_path):
                _logger.warning("Wav not found: %s", wav_path)
                continue
            emb_path = os.path.join(features_dir, f"{feature_file}.spk_emb.npy")
            if os.path.isfile(emb_path):
                continue
            signal, fs = torchaudio.load(wav_path)
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                signal = resampler(signal)
            embeddings = classifier.encode_batch(signal)
            emb_vector = embeddings.squeeze().cpu().numpy()
            np.save(emb_path, emb_vector)
            count += 1
            if count % 100 == 0:
                _logger.info("Processed %d ECAPA embeddings...", count)
    _logger.info("ECAPA: finished. Processed %d new embeddings.", count)
