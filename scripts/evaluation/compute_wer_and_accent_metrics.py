#!/usr/bin/env python3
"""Utility to score converted utterances with ASR WER and accent confidence.

This script supports two input modes:

1. Before/after comparison – provide directories that contain the same ordered
   set of WAV files (e.g., ``0_*.wav``, ``1_*.wav``). Each path starting with
   the same numeric prefix is treated as the same utterance before and after
   conversion. The script transcribes the “before” audio to obtain references
   when no transcript file is supplied.
2. Before-only analysis – provide only the source directory alongside a
   transcript file (or allow the script to transcribe the audio). This surfaces
   the baseline accent distribution/ASR quality prior to any conversion.

In both cases a text file with reference transcripts may be supplied. When it
is missing in comparison mode, the script automatically transcribes the
``before`` audio to build the references.

For every utterance, the script will:
  * Transcribe the converted audio with an OpenAI Whisper ASR checkpoint.
  * Compute the per-utterance and global (micro) WER against the transcript.
  * Run a pretrained accent classifier on the source and target
    audio to measure how the target-class confidence changes.
  * Optionally emit an overlapping histogram comparing predicted accent class
    distributions before and after conversion.

The resulting JSON file captures all per-utterance details as well as global
aggregates that can be tracked over time.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import torch
from jiwer import wer

try:  # Optional import: torchaudio is required for ASR and audio loading.
    import torchaudio
except ImportError as exc:  # pragma: no cover - informative error on demand
    torchaudio = None  # type: ignore[assignment]
    TORCHAUDIO_IMPORT_ERROR = exc
else:
    TORCHAUDIO_IMPORT_ERROR = None

try:  # Optional import: SpeechBrain provides the custom accent classifier loader.
    from speechbrain.pretrained.interfaces import foreign_class
except ImportError as exc:  # pragma: no cover - informative error on demand
    foreign_class = None  # type: ignore[assignment]


DEFAULT_WHISPER_MODEL = "medium"
ACCENT_CLASSIFIER_ID = "Jzuluaga/accent-id-commonaccent_xlsr-en-english"
ACCENT_INTERFACE_FILE = "custom_interface.py"
# Raw label order from label_encoder.txt in the HF repo.
ACCENT_DISPLAY_LABELS = [
    "us",
    "england",
    "australia",
    "indian",
    "canada",
    "bermuda",
    "scotland",
    "african",
    "ireland",
    "newzealand",
    "wales",
    "malaysia",
    "philippines",
    "singapore",
    "hongkong",
    "southatlandtic",
]
ACCENT_LABEL_OVERRIDES = {
    "us": "united_states",
    "indian": "india",
    "african": "africa",
    "newzealand": "new_zealand",
    "hongkong": "hong_kong",
    "southatlandtic": "south_atlantic",
}
ACCENT_SYNONYMS = {
    "american": "united_states",
    "us": "united_states",
    "usa": "united_states",
    "british": "england",
    "english": "england",
    "indian": "india",
    "irish": "ireland",
    "scottish": "scotland",
    "aussie": "australia",
    "african": "africa",
}
TEXT_CLEANUP_PATTERN = re.compile(r"[^a-z0-9' ]+")
_TORCH_LOAD_PATCHED = False
_FSDP_PATCHED = False


@dataclass
class AlignedUtterance:
    """Before/after audio paths that refer to the same utterance."""

    utt_id: str
    after_path: Optional[Path] = None
    before_path: Optional[Path] = None


@dataclass
class AccentPrediction:
    """Accent classifier output for a single waveform."""

    predicted_label: str
    probabilities: Dict[str, float]

    def prob_for(self, label: str) -> float:
        return self.probabilities.get(label, 0.0)


def _ensure_torchaudio_available() -> None:
    if torchaudio is None:
        raise ImportError(
            "torchaudio is required for this script but could not be imported. "
            "Install the project with the 'pytorch' extra or add torchaudio to "
            "your environment."
        ) from TORCHAUDIO_IMPORT_ERROR


def _prepare_whisper_import() -> "module":
    """Import whisper and patch torch.load for older PyTorch builds if needed."""
    global _TORCH_LOAD_PATCHED
    try:
        import whisper  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError(
            "openai-whisper is required for ASR decoding. Install it via `pip install -U openai-whisper`."
        ) from exc

    if not _TORCH_LOAD_PATCHED and "weights_only" not in inspect.signature(torch.load).parameters:
        original_load = torch.load

        def _compat_torch_load(*args, **kwargs):
            kwargs.pop("weights_only", None)
            return original_load(*args, **kwargs)

        torch.load = _compat_torch_load  # type: ignore[assignment]
        _TORCH_LOAD_PATCHED = True

    return whisper


def _ensure_fsdp_stub() -> None:
    """Create a minimal torch.distributed.fsdp stub if PyTorch lacks it."""
    global _FSDP_PATCHED
    if _FSDP_PATCHED:
        return
    try:
        import torch.distributed.fsdp  # type: ignore
        _FSDP_PATCHED = True
        return
    except ModuleNotFoundError:
        pass
    import types
    import sys

    dist_module = sys.modules.get("torch.distributed")
    if dist_module is None:
        dist_module = types.ModuleType("torch.distributed")
        sys.modules["torch.distributed"] = dist_module
    fsdp_module = types.ModuleType("torch.distributed.fsdp")

    class _DummyFSDP:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("FSDP is unavailable in this PyTorch build.")

    fsdp_module.FullyShardedDataParallel = _DummyFSDP  # type: ignore[attr-defined]
    setattr(dist_module, "fsdp", fsdp_module)
    sys.modules["torch.distributed.fsdp"] = fsdp_module
    _FSDP_PATCHED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute WER and accent confidence deltas for converted audio."
    )
    parser.add_argument(
        "--before-dir",
        type=Path,
        required=True,
        help="Directory containing the source (pre-conversion) WAV files.",
    )
    parser.add_argument(
        "--after-dir",
        type=Path,
        default=None,
        help="Optional directory containing the converted (post-conversion) WAV files. "
        "At least one of --before-dir/--after-dir must be provided.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("conversion_metrics.json"),
        help="Where to store the aggregated metrics JSON (default: conversion_metrics.json).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional path to save an overlapping histogram of predicted accent classes.",
    )
    parser.add_argument(
        "--transcript-file",
        type=Path,
        default=None,
        help="Optional reference transcript file (one line per utterance, matching the sorted WAV order). "
        "If omitted, the script will transcribe the before-audio to build the reference text.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference (default: auto-detect).",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help="OpenAI Whisper checkpoint to use for ASR decoding (e.g., tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--accent-cache-dir",
        type=Path,
        default=Path(".cache") / "accent_classifier",
        help="Directory where the accent model files will be stored (default: .cache/accent_classifier).",
    )
    parser.add_argument(
        "--source-class",
        default=None,
        help="Accent label that represents the source/domain to be reduced. "
        "Required when --after-dir is provided.",
    )
    parser.add_argument(
        "--target-class",
        required=True,
        help="Accent label that represents the desired target domain.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places to keep in the JSON (default: 4).",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=None,
        help="Optional cap on how many utterances to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--disable-accent-cache",
        action="store_true",
        help="Recompute accent predictions even if the same audio path appears multiple times.",
    )
    parser.add_argument(
        "--experiment-name",
        default="",
        help="Optional subdirectory created under the output/plot paths in which artifacts will be stored.",
    )
    return parser.parse_args()


def _normalize_label(label: str) -> str:
    return label.strip().lower()


def _normalize_text(text: str) -> List[str]:
    cleaned = TEXT_CLEANUP_PATTERN.sub(" ", text.lower())
    return [token for token in cleaned.split() if token]


def _normalized_text_string(text: str) -> str:
    return " ".join(_normalize_text(text))


def _apply_experiment_dir(path: Optional[Path], experiment: str) -> Optional[Path]:
    if not experiment or path is None:
        return path
    return path.parent / experiment / path.name


def _collect_wav_files(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    if not directory.is_dir():
        raise ValueError(f"Path '{directory}' is not a directory.")
    wavs = sorted(directory.rglob("*.wav"), key=lambda p: p.name)
    if not wavs:
        raise ValueError(f"No WAV files found inside '{directory}'.")
    return wavs


def _alignment_key(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def align_audio_directories(after_dir: Optional[Path], before_dir: Optional[Path]) -> List[AlignedUtterance]:
    if after_dir is None and before_dir is None:
        raise ValueError("At least one of --before-dir/--after-dir must be provided.")
    aligned: List[AlignedUtterance] = []
    seen_ids = set()

    if after_dir is not None and before_dir is not None:
        after_wavs = _collect_wav_files(after_dir)
        before_wavs = _collect_wav_files(before_dir)
        if len(before_wavs) != len(after_wavs):
            raise ValueError(
                f"Directories do not contain the same number of WAV files: "
                f"{before_dir} has {len(before_wavs)} while {after_dir} has {len(after_wavs)}."
            )
        for idx, (before_path, after_path) in enumerate(zip(before_wavs, after_wavs)):
            before_key = _alignment_key(before_path)
            after_key = _alignment_key(after_path)
            if before_key != after_key:
                raise ValueError(
                    "Sorted WAV files do not align. "
                    f"Encountered '{before_path.name}' and '{after_path.name}' at position {idx}."
                )
            utt_id = before_key or f"{idx}"
            if utt_id in seen_ids:
                raise ValueError(f"Duplicate utterance id '{utt_id}' derived from '{before_path.name}'.")
            seen_ids.add(utt_id)
            aligned.append(
                AlignedUtterance(
                    utt_id=utt_id,
                    after_path=after_path.resolve(),
                    before_path=before_path.resolve(),
                )
            )
        return aligned

    if after_dir is not None:
        after_wavs = _collect_wav_files(after_dir)
        for idx, after_path in enumerate(after_wavs):
            utt_id = _alignment_key(after_path) or f"{idx}"
            if utt_id in seen_ids:
                raise ValueError(
                    f"Duplicate utterance id '{utt_id}' derived from '{after_path.name}'."
                )
            seen_ids.add(utt_id)
            aligned.append(
                AlignedUtterance(
                    utt_id=utt_id,
                    after_path=after_path.resolve(),
                    before_path=None,
                )
            )
        return aligned

    if before_dir is None:
        raise ValueError("before_dir must be provided when --after-dir is omitted.")
    before_wavs = _collect_wav_files(before_dir)  # after_dir is None here
    for idx, before_path in enumerate(before_wavs):
        utt_id = _alignment_key(before_path) or f"{idx}"
        if utt_id in seen_ids:
            raise ValueError(
                f"Duplicate utterance id '{utt_id}' derived from '{before_path.name}'."
            )
        seen_ids.add(utt_id)
        aligned.append(
            AlignedUtterance(
                utt_id=utt_id,
                after_path=None,
                before_path=before_path.resolve(),
            )
        )
    return aligned


def load_transcript_file(path: Path, expected_count: int) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if len(lines) != expected_count:
        raise ValueError(
            f"Transcript file '{path}' contains {len(lines)} usable lines but {expected_count} utterances were detected."
        )
    return lines


class WhisperASR:
    """Minimal ASR wrapper that always decodes with OpenAI Whisper."""

    def __init__(self, model_id: str, device: str) -> None:
        _ensure_torchaudio_available()
        self.device = torch.device(device)
        whisper = _prepare_whisper_import()
        self.model = whisper.load_model(model_id, device=device)
        self.sample_rate = 16000

    def transcribe(self, audio_path: Path) -> str:
        waveform, sr = torchaudio.load(str(audio_path))
        waveform = self._to_mono(waveform)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        audio = waveform.squeeze(0).cpu().numpy()
        result = self.model.transcribe(audio, fp16=(self.device.type == "cuda"))
        return str(result.get("text", "")).strip()

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.size(0) == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)


class SpeechBrainAccentClassifier:
    """Accent classifier wrapper built on SpeechBrain foreign_class helper."""

    def __init__(self, cache_dir: Path, device: str) -> None:
        if foreign_class is None:
            raise ImportError(
                "speechbrain is required for accent classification. Install it via `pip install speechbrain`."
            )
        _ensure_fsdp_stub()
        self.device = torch.device(device)
        cache_dir.mkdir(parents=True, exist_ok=True)
        run_opts = {"device": device}
        self.classifier = foreign_class(
            source=ACCENT_CLASSIFIER_ID,
            pymodule_file=ACCENT_INTERFACE_FILE,
            classname="CustomEncoderWav2vec2Classifier",
            savedir=str(cache_dir),
            run_opts=run_opts,
        )
        self.sample_rate = 16000
        label_encoder = getattr(self.classifier.hparams, "label_encoder", None)
        labels: List[str] = []
        if label_encoder is not None and hasattr(label_encoder, "ind2lab"):
            try:
                indices = sorted(label_encoder.ind2lab.keys())  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - defensive
                indices = list(label_encoder.ind2lab.keys())  # type: ignore[attr-defined]
            labels = [str(label_encoder.ind2lab[i]) for i in indices]  # type: ignore[index]
        if not labels:
            labels = list(ACCENT_DISPLAY_LABELS)
        self.labels = labels
        self.display_map = self._build_display_map(self.labels)

    def predict(self, audio_path: Path) -> AccentPrediction:
        waveform, sr = torchaudio.load(str(audio_path))
        waveform = self._to_mono(waveform)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.to(self.device)
        with torch.inference_mode():
            outputs = self.classifier.classify_batch(waveform)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.squeeze(0)
        else:
            # SpeechBrain custom interface returns tuple of (posteriors, score, index, label)
            posteriors = outputs[0]
            logits = posteriors.squeeze(0)
        probabilities = logits # This already sums to 1, no need in softmax #torch.softmax(logits, dim=-1)
        # Assert if sums to 1
        assert torch.isclose(probabilities.sum(), torch.tensor(1.0)), "Probabilities do not sum to 1"
        probs_by_label = {
            self.labels[i]: float(probabilities[i].item())
            for i in range(min(len(self.labels), probabilities.numel()))
        }
        predicted_idx = int(torch.argmax(probabilities).item())
        predicted_label = self.labels[predicted_idx]
        return AccentPrediction(predicted_label=predicted_label, probabilities=probs_by_label)

    def display_label(self, raw_label: str) -> str:
        key = _normalize_label(raw_label)
        return self.display_map.get(key, raw_label)

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.size(0) == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)

    def _build_display_map(self, labels: List[str]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for idx, label in enumerate(labels):
            if label in ACCENT_LABEL_OVERRIDES:
                display = ACCENT_LABEL_OVERRIDES[label]
            elif idx < len(ACCENT_DISPLAY_LABELS):
                candidate = ACCENT_DISPLAY_LABELS[idx]
                display = ACCENT_LABEL_OVERRIDES.get(candidate, candidate)
            else:
                display = label
            mapping[_normalize_label(label)] = display
            mapping[_normalize_label(display)] = display
        for alias, target in ACCENT_SYNONYMS.items():
            normalized_target = _normalize_label(target)
            if normalized_target in mapping:
                mapping[_normalize_label(alias)] = mapping[normalized_target]
        return mapping


def main() -> None:
    args = parse_args()
    args.output_json = _apply_experiment_dir(args.output_json, args.experiment_name)
    args.plot_path = _apply_experiment_dir(args.plot_path, args.experiment_name)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    has_before = True  # argparse enforces before_dir
    has_after = args.after_dir is not None

    aligned_pairs = align_audio_directories(args.after_dir, args.before_dir)

    transcript_lines: Optional[List[str]] = None
    if args.transcript_file is not None:
        transcript_lines = load_transcript_file(args.transcript_file, len(aligned_pairs))

    if args.max_utterances is not None:
        aligned_pairs = aligned_pairs[: args.max_utterances]
        if transcript_lines is not None:
            transcript_lines = transcript_lines[: args.max_utterances]

    asr = WhisperASR(model_id=args.whisper_model, device=device.type)
    accent_classifier = SpeechBrainAccentClassifier(
        cache_dir=args.accent_cache_dir,
        device=device.type,
    )

    def _format_class_distribution(prediction: AccentPrediction) -> Dict[str, float]:
        distribution: Dict[str, float] = OrderedDict()
        for raw_label in accent_classifier.labels:
            display = accent_classifier.display_label(raw_label)
            distribution[display] = round(prediction.probabilities.get(raw_label, 0.0), args.precision)
        for raw_label, prob in prediction.probabilities.items():
            if raw_label not in accent_classifier.labels:
                display = accent_classifier.display_label(raw_label)
                if display not in distribution:
                    distribution[display] = round(prob, args.precision)
        return distribution

    normalized_labels = {_normalize_label(label): label for label in accent_classifier.labels}
    for raw_label in accent_classifier.labels:
        display_name = accent_classifier.display_label(raw_label)
        normalized_labels.setdefault(_normalize_label(display_name), raw_label)
    if _normalize_label(args.target_class) not in normalized_labels:
        raise ValueError(
            f"Target class '{args.target_class}' not found in accent classifier labels: {accent_classifier.labels}"
        )
    target_label = normalized_labels[_normalize_label(args.target_class)]

    source_label: Optional[str] = None
    if args.source_class is not None:
        normalized_source = _normalize_label(args.source_class)
        if normalized_source not in normalized_labels:
            raise ValueError(
                f"Source class '{args.source_class}' not found in accent classifier labels: {accent_classifier.labels}"
            )
        source_label = normalized_labels[normalized_source]
    elif has_after:
        raise ValueError("--source-class is required whenever --after-dir is provided.")

    per_utt = OrderedDict()
    total_word_errors = 0
    total_ref_words = 0
    all_refs: List[str] = []
    all_hyps: List[str] = []
    before_target_values: List[float] = []
    after_target_values: List[float] = []
    before_source_values: List[float] = []
    after_source_values: List[float] = []
    before_class_predictions: List[str] = []
    after_class_predictions: List[str] = []
    accent_cache: Dict[str, AccentPrediction] = {}
    prob_sums_before: Optional[Dict[str, float]] = (
        {label: 0.0 for label in accent_classifier.labels} if has_before else None
    )
    prob_sums_after: Optional[Dict[str, float]] = (
        {label: 0.0 for label in accent_classifier.labels} if has_after else None
    )
    before_utt_count = 0
    after_utt_count = 0

    def _get_accent(path: Path) -> AccentPrediction:
        key = str(path)
        if args.disable_accent_cache or key not in accent_cache:
            accent_cache[key] = accent_classifier.predict(path)
        return accent_cache[key]

    for idx, pair in enumerate(aligned_pairs):
        utt_id = pair.utt_id
        if transcript_lines is not None:
            reference_transcript = transcript_lines[idx]
        else:
            raise RuntimeError("Reference transcript is unavailable for this evaluation.")

        eval_audio_path = pair.after_path or pair.before_path
        if eval_audio_path is None:
            raise RuntimeError(f"No audio paths found for utterance '{utt_id}'.")

        hypothesis = asr.transcribe(eval_audio_path)

        normalized_ref = _normalized_text_string(reference_transcript)
        normalized_hyp = _normalized_text_string(hypothesis)
        all_refs.append(normalized_ref)
        all_hyps.append(normalized_hyp)
        ref_tokens = len(_normalize_text(reference_transcript))
        ref_len = ref_tokens
        utt_wer = wer(normalized_ref or "", normalized_hyp or "") if ref_len > 0 else 0.0
        word_errors = int(round(utt_wer * ref_len))
        total_word_errors += word_errors
        total_ref_words += ref_tokens

        before_accent: Optional[AccentPrediction] = None
        if pair.before_path is not None:
            before_accent = _get_accent(pair.before_path)
            before_class_predictions.append(before_accent.predicted_label)
            before_utt_count += 1
        after_accent: Optional[AccentPrediction] = None
        if pair.after_path is not None:
            after_accent = _get_accent(pair.after_path)
            after_class_predictions.append(after_accent.predicted_label)
            after_utt_count += 1

        before_target = before_accent.prob_for(target_label) if before_accent else None
        after_target = after_accent.prob_for(target_label) if after_accent else None
        if before_target is not None:
            before_target_values.append(before_target)
        if after_target is not None:
            after_target_values.append(after_target)

        before_source = (
            before_accent.prob_for(source_label) if before_accent and source_label else None
        )
        after_source = (
            after_accent.prob_for(source_label) if after_accent and source_label else None
        )
        if before_source is not None:
            before_source_values.append(before_source)
        if after_source is not None:
            after_source_values.append(after_source)

        for label in accent_classifier.labels:
            if prob_sums_before is not None and before_accent is not None:
                prob_sums_before[label] += before_accent.probabilities.get(label, 0.0)
            if prob_sums_after is not None and after_accent is not None:
                prob_sums_after[label] += after_accent.probabilities.get(label, 0.0)

        accent_delta = (
            (after_target - before_target)
            if (after_target is not None and before_target is not None)
            else None
        )
        audio_paths: Dict[str, str] = {}
        if pair.before_path is not None:
            audio_paths["before"] = str(pair.before_path)
        if pair.after_path is not None:
            audio_paths["after"] = str(pair.after_path)
        accent_details: Dict[str, Dict[str, object]] = {}
        if before_accent is not None and before_target is not None:
            before_entry = {
                "predicted": accent_classifier.display_label(before_accent.predicted_label),
                "target_confidence": round(before_target, args.precision),
                "class_distribution": _format_class_distribution(before_accent),
            }
            if before_source is not None:
                before_entry["source_confidence"] = round(before_source, args.precision)
            accent_details["before"] = before_entry
        if after_accent is not None and after_target is not None:
            after_entry = {
                "predicted": accent_classifier.display_label(after_accent.predicted_label),
                "target_confidence": round(after_target, args.precision),
                "class_distribution": _format_class_distribution(after_accent),
            }
            if after_source is not None:
                after_entry["source_confidence"] = round(after_source, args.precision)
            accent_details["after"] = after_entry
        if accent_delta is not None:
            accent_details["target_confidence_delta"] = round(accent_delta, args.precision)

        per_utt[utt_id] = {
            "reference_transcript": reference_transcript,
            "asr_transcript": hypothesis,
            "normalized_transcripts": {
                "reference": normalized_ref,
                "asr": normalized_hyp,
            },
            "audio_paths": audio_paths,
            "wer": round(utt_wer, args.precision),
            "word_errors": word_errors,
            "reference_word_count": ref_len,
            "accent": accent_details,
        }

    micro_wer = wer(all_refs, all_hyps) if all_refs else 0.0

    def _avg(values: List[float]) -> float:
        return float(mean(values)) if values else 0.0

    accent_summary: Dict[str, object] = {
        "target_class": accent_classifier.display_label(target_label),
    }
    if after_target_values:
        accent_summary["after_avg_target_conf"] = round(_avg(after_target_values), args.precision)
    if before_target_values:
        accent_summary["before_avg_target_conf"] = round(_avg(before_target_values), args.precision)
    if before_target_values and after_target_values:
        accent_summary["avg_target_conf_delta"] = round(
            _avg(after_target_values) - _avg(before_target_values), args.precision
        )

    source_label_display: Optional[str] = None
    if source_label:
        source_label_display = accent_classifier.display_label(source_label)
        accent_summary["source_class"] = source_label_display
        if after_source_values:
            accent_summary["after_avg_source_conf"] = round(_avg(after_source_values), args.precision)
        if before_source_values:
            accent_summary["before_avg_source_conf"] = round(_avg(before_source_values), args.precision)

    if has_before and before_class_predictions:
        before_counts = Counter(before_class_predictions)
        accent_summary["predicted_distribution_before"] = {
            accent_classifier.display_label(label): count for label, count in before_counts.items()
        }

    if after_class_predictions:
        after_counts = Counter(after_class_predictions)
        accent_summary["predicted_distribution_after"] = {
            accent_classifier.display_label(label): count for label, count in after_counts.items()
        }

    avg_conf_before: Optional[Dict[str, float]] = None
    if prob_sums_before is not None and before_utt_count > 0:
        avg_conf_before = {
            label: prob_sums_before[label] / before_utt_count for label in accent_classifier.labels
        }
    avg_conf_after: Optional[Dict[str, float]] = None
    if prob_sums_after is not None and after_utt_count > 0:
        avg_conf_after = {
            label: prob_sums_after[label] / after_utt_count for label in accent_classifier.labels
        }
    if avg_conf_before is not None:
        accent_summary["confidence_distribution_before"] = {
            accent_classifier.display_label(label): round(avg_conf_before[label], args.precision)
            for label in accent_classifier.labels
        }
    if avg_conf_after is not None:
        accent_summary["confidence_distribution_after"] = {
            accent_classifier.display_label(label): round(avg_conf_after[label], args.precision)
            for label in accent_classifier.labels
        }

    metrics = {
        "global": {
            "micro_wer": round(micro_wer, args.precision),
            "total_word_errors": total_word_errors,
            "total_reference_words": total_ref_words,
            "utterance_count": len(per_utt),
            "whisper_model": args.whisper_model,
        },
        "accent_summary": accent_summary,
        "per_utterance": per_utt,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as writer:
        json.dump(metrics, writer, indent=2, ensure_ascii=False)

    if args.plot_path and (avg_conf_before is not None or avg_conf_after is not None):
        save_histogram_plot(
            avg_conf_before,
            avg_conf_after,
            accent_classifier,
            out_path=args.plot_path,
            source_label=source_label_display,
            target_label=accent_summary["target_class"],
        )

    print(
        f"Saved metrics for {len(per_utt)} utterances. Global micro WER: {metrics['global']['micro_wer']:.{args.precision}f}."
    )


def save_histogram_plot(
    before_avg_conf: Optional[Dict[str, float]],
    after_avg_conf: Optional[Dict[str, float]],
    classifier: SpeechBrainAccentClassifier,
    out_path: Path,
    source_label: Optional[str],
    target_label: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    ordered_raw_labels = list(classifier.labels)
    combined_keys: set[str] = set()
    if after_avg_conf is not None:
        combined_keys |= set(after_avg_conf.keys())
    if before_avg_conf is not None:
        combined_keys |= set(before_avg_conf.keys())
    extra_labels = [label for label in sorted(combined_keys) if label not in ordered_raw_labels]
    ordered_raw_labels.extend(extra_labels)
    if not ordered_raw_labels:
        return
    display_labels = [classifier.display_label(label) for label in ordered_raw_labels]
    positions = np.arange(len(ordered_raw_labels))

    plt.figure(figsize=(max(6, len(ordered_raw_labels) * 1.2), 4))
    plotted = False
    if before_avg_conf is not None:
        before_values = np.array([before_avg_conf.get(label, 0.0) for label in ordered_raw_labels])
        before_label = source_label or "before"
        plt.bar(
            positions,
            before_values,
            label=f"Before ({before_label})",
            color="#1f77b4",
            alpha=0.6,
        )
        plotted = True
    if after_avg_conf is not None:
        after_values = np.array([after_avg_conf.get(label, 0.0) for label in ordered_raw_labels])
        plt.bar(
            positions,
            after_values,
            label=f"After ({target_label})",
            color="#ff7f0e",
            alpha=0.6 if before_avg_conf is not None else 0.8,
        )
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xticks(positions, display_labels, rotation=30, ha="right")
    plt.ylabel("Average Confidence")
    if before_avg_conf is not None and after_avg_conf is not None:
        plt.title("Accent Class Distribution Before vs After Conversion")
    elif after_avg_conf is not None:
        plt.title("Accent Class Distribution After Conversion")
    else:
        plt.title("Accent Class Distribution Before Conversion")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    main()
