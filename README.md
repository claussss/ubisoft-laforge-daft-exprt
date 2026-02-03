<!-- omit in toc -->
# Daft-Exprt: Cross-Speaker Prosody Transfer on Any Text for Expressive Speech Synthesis

<!-- omit in toc -->
## Table of Contents
- [Installation](#installation)
  - [Local Environment](#local-environment)
  - [Docker Image](#docker-image)
- [Quick Start](#quick-start)
  - [Dataset format](#dataset-format)
  - [Format datasets (LJ / ESD)](#format-datasets-lj--esd)
  - [Pre-processing](#pre-processing)
  - [Training](#training)
  - [Vocoder fine-tuning dataset](#vocoder-fine-tuning-dataset)
  - [Synthesis](#synthesis)
- [Script reference](#script-reference)
  - [training.py](#trainingpy)
  - [extract_symbol_prosody.py](#extract_symbol_prosodypy)
  - [compute_spk_stats_from_prosody.py](#compute_spk_stats_from_prosodypy)
  - [adapt_accent.py](#adapt_accentpy)
  - [synthesize.py](#synthesizepy)
  - [format_dataset.py](#format_datasetpy)
  - [Grid search and evaluation](#grid-search-and-evaluation)
- [Citation](#citation)
- [Contributing](#contributing)

## Installation

### Local Environment

Requirements:
- Ubuntu >= 20.04
- Python >= 3.8
- [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx?lang=en-us) >= 450.80.02
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) >= 11.1
- [CuDNN](https://developer.nvidia.com/rdp/cudnn-archive) >= v8.0.5

We recommend using conda. Create the environment and install dependencies:

1. `conda create -n daft_exprt python=3.8 -y`
2. `conda activate daft_exprt`
3. `cd environment`
4. `make`

The repository is installed as a pip package in editable mode. Scripts live in `scripts/`; core code in `src/daft_exprt/`. Config defaults are in `src/daft_exprt/hparams.py`.

### Docker Image

Requirements: [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker), NVIDIA Driver >= 450.80.02.

Build: `docker build -f environment/Dockerfile -t daft_exprt .`

---

## 🚀 Quick Start

---

### Dataset format

Each speaker dataset must follow this layout:

```
/speaker_dir
    metadata.csv
    /wavs
        wav_file_name_1.wav
        ...
```

`metadata.csv` format (pipe `|` separator):

```
wav_file_name_1|text_1
wav_file_name_N|text_N
```

All speaker roots must sit under a single data root, e.g.:

```
/data_dir
    LJ_Speech
    ESD
        spk_1
        ...
```

### Format datasets (LJ / ESD)

Use `scripts/format_dataset.py` to produce the above structure for LJ and ESD:

```bash
python scripts/format_dataset.py --data_set_dir /path/to/LJ_Speech LJ
python scripts/format_dataset.py --data_set_dir /path/to/ESD ESD --language english
```

### Pre-processing

Pre-processing runs MFA alignment, feature extraction, train/validation split, feature stats, and **ECAPA speaker embeddings**.

```bash
python scripts/training.py \
    --experiment_name EXPERIMENT_NAME \
    --data_set_dir /path/to/data_dir \
    pre_process
```

**Where outputs go:**

- **Extracted features** (`.npy`, etc.) → features directory: by default `datasets/<language>/<sampling_rate>Hz/` (e.g. `datasets/english/22050Hz/`), or the path you pass to `--features_dir`.
- **Experiment bookkeeping** (train/validation file lists, `config.json`, `stats.json`, logs, checkpoints) → `trainings/EXPERIMENT_NAME/`.

To restrict to specific speakers:

```bash
python scripts/training.py \
    --experiment_name EXPERIMENT_NAME \
    --speakers ESD/spk_1 ESD/spk_2 \
    --data_set_dir /path/to/data_dir \
    pre_process
```

### Training

Train on all pre-processed speakers (or the same `--speakers` list used at pre-process):

```bash
python scripts/training.py \
    --experiment_name EXPERIMENT_NAME \
    --data_set_dir /path/to/data_dir \
    train
```

> **Note:** During training, `train.py` does **not** load data from `data_set_dir`.  
> Instead, it reads training and validation file lists from:
>
> - `trainings/<experiment>/train_english.txt`
> - `trainings/<experiment>/validation_english.txt`

Resume from a checkpoint with `--checkpoint CHECKPOINT_PATH`. TensorBoard: `tensorboard --logdir trainings/EXPERIMENT_NAME/logs`.

### Vocoder fine-tuning dataset

The `fine_tune` command **generates a mel + wav dataset** for training a vocoder (e.g. HiFi-GAN) **outside this repository**. This repo does not include vocoder training code; it only produces the paired data.

```bash
python scripts/training.py \
    --experiment_name EXPERIMENT_NAME \
    --data_set_dir /path/to/data_dir \
    fine_tune \
    --checkpoint CHECKPOINT_PATH
```

Output: `trainings/EXPERIMENT_NAME/fine_tuning_dataset/`.

### Synthesis

Synthesis requires:

- **Symbol-level prosody file** — one line per utterance: list of `(symbol, duration, pitch, energy)` tuples (e.g. from `extract_symbol_prosody.py`).
- **Speaker stats** — JSON with mean/std or a directory of wavs to compute them.
- **Speaker and accent embeddings** — from audio dirs (`--spk_emb_audios_dir`, `--accent_emb_audios_dir`), or from the checkpoint (`memorized_spk_emb` / `memorized_accent_emb`, e.g. after `adapt_accent.py`). If neither source yields both, the script exits with an error.
- **Vocoder** — HiFi-GAN; a universal model is used if `--vocoder_checkpoint` is omitted (downloaded on demand).

**Example:**

```bash
python scripts/synthesize.py \
    --output_dir /path/to/outputs \
    --checkpoint trainings/MY_EXP/checkpoints/checkpoint.pt \
    --symbol_prosody_file /path/to/prosody.txt \
    --new_speaker_stats /path/to/stats.json \
    --spk_emb_audios_dir /path/to/speaker_wavs \
    --accent_emb_audios_dir /path/to/accent_wavs
```

If the checkpoint already contains memorized embeddings (e.g. from accent adaptation), you can omit the audio dirs.

---

## 📋 Script reference

---

### training.py

Entry point for pre-processing, training, and vocoder fine-tuning dataset generation. All commands share:

| Argument | Default | Description |
|----------|---------|-------------|
| `-en, --experiment_name` | — | Name of the experiment directory under `trainings/`. |
| `-dd, --data_set_dir` | — | Root path containing speaker datasets. |
| `-spks, --speakers` | `[]` | Optional list of speaker subdirs; if empty, all speakers under `data_set_dir` are used. |
| `-lg, --language` | `'english'` | Language for MFA/text. |
| `-cfg, --config_file` | `''` | Optional config path; default is `trainings/<experiment_name>/config.json`. |

**pre_process**

| Argument | Default | Description |
|----------|---------|-------------|
| `-fd, --features_dir` | `PROJECT_ROOT/datasets` | Where to store pre-processed features. When this default is used, features are written to `datasets/<language>/<sampling_rate>Hz/` (e.g. `datasets/english/22050Hz/`). |
| `-pv, --proportion_validation` | `0.1` | Proportion (e.g. 0.1 = 10%) of each speaker’s data used for validation. |
| `-nj, --nb_jobs` | `'6'` | Number of jobs for multiprocessing; `'max'` uses all cores. |

**train**

| Argument | Default | Description |
|----------|---------|-------------|
| `-chk, --checkpoint` | `''` | Path to checkpoint to resume from. |
| `-nmpd, --no_multiprocessing_distributed` | `False` | Disable distributed training. |
| `-ws, --world_size` | `1` | Number of nodes. |
| `-r, --rank` | `0` | Node rank. |
| `-m, --master` | `'tcp://localhost:54321'` | Master URL for distributed training. |
| `--cpu` | `False` | Use CPU (e.g. for debugging). |

**fine_tune**

| Argument | Default | Description |
|----------|---------|-------------|
| `-chk, --checkpoint` | — | Checkpoint used to generate mel/wav pairs for external vocoder training. |

---

### extract_symbol_prosody.py

Extracts symbol-level duration, pitch, and energy from audio using a manifest, MFA alignment, and REAPER pitch. Manifest format: one line per audio, `absolute_path_to_audio.wav|transcript` (separator is fixed as `|`). Output: one line per audio, each line a Python list of `(symbol, duration_frames, pitch, energy)` tuples.

**Dependencies**: `mfa` CLI, MFA acoustic/G2P models (`mfa download ...`), `reaper` on PATH.

| Argument | Default | Description |
|----------|---------|-------------|
| `--manifest` | — | Text file with one `audio_path\|transcript` per line. |
| `--output` | — | Output txt path. |
| `--language` | `'english'` | Language for MFA and text cleaner. |
| `--g2p_preset` | `'american_english'` | Preset for G2P (`american_english` or `indian_english`). |
| `--g2p_model` | `''` | Explicit path to MFA G2P model zip (overrides preset). |
| `--dictionary` | `''` | MFA pronunciation dictionary; defaults to config default. |
| `--acoustic_model` | `''` | MFA acoustic model zip; defaults to config default. |
| `--nb_jobs` | `4` | Parallel jobs for MFA. |
| `--work_dir` | `PROJECT_ROOT/tmp_symbol_prosody` | Scratch directory. |
| `--keep_temps` | `False` | Keep intermediate files. |

Example:

```bash
python scripts/extract_symbol_prosody.py \
    --manifest /path/to/list.txt \
    --output /path/to/prosody.txt \
    --g2p_preset american_english \
    --nb_jobs 4
```

Output lines look like: `[('DH', 5, 3.512, 1.104), ('AH0', 4, 3.403, 1.089), ...]`

---

### compute_spk_stats_from_prosody.py

Computes pitch and energy mean/std from a symbol-prosody file (e.g. produced by `extract_symbol_prosody.py`). Used to generate `--new_speaker_stats` for synthesis when you do not have training-set stats.

**Input:** One line per utterance; each line is a Python literal: either a list of `(symbol, duration, pitch, energy)` tuples (extract_symbol_prosody format) or a single tuple `(symbols, durations, pitch, energy)` with four sequences. Only voiced pitch (`> 0`) and non-zero energy values are included.

**Output:** JSON with `pitch` and `energy`, each having `mean` and `std` (population std).

| Argument | Default | Description |
|----------|---------|-------------|
| `input_txt` | — | File produced by `extract_symbol_prosody.py` (or same format). |
| `--output` | — | Optional JSON path; if omitted, stats are printed to stdout. |

Example:

```bash
python scripts/compute_spk_stats_from_prosody.py prosody.txt --output stats.json
```

---

### adapt_accent.py

Few-shot accent adaptation by unfreezing (all or selected) decoder blocks and optimizing on accent data. Saves checkpoints with `memorized_spk_emb` and `memorized_accent_emb` for use in synthesis.

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | — | Base Daft-Exprt checkpoint. |
| `--config_file` | — | Config JSON path. |
| `--accent_dir` | — | Directory of accent feature `.npy` files (and optional `.spk_emb.npy`). |
| `--recon_dir` | `''` | Directory for reconstruction targets; defaults to `accent_dir`. |
| `--output_name` | — | Name for the adapted checkpoint. |
| `--learning_rate` | `0.01` | Learning rate. |
| `--iterations` | `100` | Training steps. |
| `--unfreeze_decoder_blocks` | `''` | Comma-separated block indices or `all`/`none`. |
| `--unfreeze_decoder_projection` | `False` | Unfreeze decoder projection. |
| `--unfreeze_decoder_blocks_ff_only` | `False` | Unfreeze only feed-forward in blocks. |
| `--unfreeze_decoder_blocks_att_only` | `False` | Unfreeze only attention in blocks. |
| `--unfreeze_decoder_blocks_convs_only` | `False` | Unfreeze only convs in blocks. |
| `--seed` | `42` | Random seed. |
| `--lambda_delta_l1` | `0.0` | L1 penalty on parameter deltas. |
| `--stats_path` | `''` | Path to stats JSON for normalization. |
| `--random_spk_emb` | `False` | Use random speaker embedding. |
| `--mean_spk_emb` | `False` | Use mean speaker embedding. |
| `--random_accent_emb` | `False` | Use random accent embedding. |
| `--mean_accent_emb` | `False` | Use mean accent embedding. |

---

### synthesize.py

Generates speech from a symbol-prosody file and speaker/accent conditioning. Requires a HiFi-GAN vocoder.

| Argument | Default | Description |
|----------|---------|-------------|
| `-out, --output_dir` | — | Directory for output wavs and plots. |
| `-chk, --checkpoint` | — | Daft-Exprt checkpoint (may contain memorized embeddings). |
| `-spf, --symbol_prosody_file` | — | File with one line per utterance: list of `(symbol, duration, pitch, energy)` tuples. |
| `--vocoder_checkpoint` | `''` | HiFi-GAN generator path; if empty, universal model is used. |
| `-bs, --batch_size` | `50` | Batch size for inference. |
| `--new_speaker_stats` | — | JSON stats path or directory of wavs to compute stats. |
| `--plot_prosody_files_to_compare` | `''` | Optional manifest of wavs for prosody comparison plots. |
| `--alpha_dur` | `1.0` | Duration scaling. |
| `--alpha_pitch` | `1.0` | Pitch scaling. |
| `--alpha_energy` | `1.0` | Energy scaling. |
| `--spk_emb_audios_dir` | `''` | Directory of wavs to compute speaker embedding. |
| `--accent_emb_audios_dir` | `''` | Directory of wavs to compute accent embedding. |

**Embeddings:** From audio dirs first; if missing, from `memorized_spk_emb` / `memorized_accent_emb` in the checkpoint. If neither source yields both, the script exits with an error.

---

### format_dataset.py

Formats LJ Speech and ESD into the required directory layout. Usage:

```bash
python scripts/format_dataset.py --data_set_dir /path/to/LJ_Speech LJ
python scripts/format_dataset.py --data_set_dir /path/to/ESD ESD --language english
```

---

### Grid search and evaluation

- **Grid search**: `scripts/grid_search_accent_v2.py` runs multiple `adapt_accent.py` runs with different hyperparameters (edit the script’s config block for paths and flags).
- **WER and accent metrics**: `scripts/evaluation/compute_wer_and_accent_metrics.py` compares pre- and post-conversion manifests with Whisper and SpeechBrain accent ID; see script help for arguments.

---