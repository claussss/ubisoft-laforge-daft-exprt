<!-- omit in toc -->
# Daft-Exprt: Cross-Speaker Prosody Transfer on Any Text for Expressive Speech Synthesis

<!-- omit in toc -->
### Julian Zaïdi, Hugo Seuté, Benjamin van Niekerk, Marc-André Carbonneau
In our [paper](https://arxiv.org/abs/2108.02271) we propose Daft-Exprt, a multi-speaker acoustic model advancing the state-of-the-art for cross-speaker prosody transfer on any text. This is one of the most challenging, and rarely directly addressed, task in speech synthesis, especially for highly expressive data. Daft-Exprt uses FiLM conditioning layers to strategically inject different prosodic information in all parts of the architecture. The model explicitly encodes traditional low-level prosody features such as pitch, loudness and duration, but also higher level prosodic information that helps generating convincing voices in highly expressive styles. Speaker identity and prosodic information are disentangled through an adversarial training strategy that enables accurate prosody transfer across speakers. Experimental results show that Daft-Exprt significantly outperforms strong baselines on inter-text cross-speaker prosody transfer tasks, while yielding naturalness comparable to state-of-the-art expressive models. Moreover, results indicate that the model discards speaker identity information from the prosody representation, and consistently generate speech with the desired voice. Visit our [demo page](https://ubisoft-laforge.github.io/speech/daft-exprt/) for audio samples related to the paper experiments.  

### Pre-trained model
**Full disclosure**: The model provided in this repository is not the same as in the paper evaluation. The model of the paper was trained with proprietary data which prevents us to release it publicly.  
We pre-train Daft-Exprt on a combination of [LJ speech dataset](https://keithito.com/LJ-Speech-Dataset/) and the emotional speech dataset (ESD) from [Zhou et al](https://github.com/HLTSingapore/Emotional-Speech-Data).  
Visit the [releases](https://github.com/ubisoft/ubisoft-laforge-daft-exprt/releases) of this repository to download the pre-trained model and to listen to prosody transfer examples using this same model.  

<!-- omit in toc -->
## Table of Contents
- [Installation](#installation)
  - [Local Environment](#local-environment)
  - [Docker Image](#docker-image)
- [Quick Start Example](#quick-start-example)
  - [Introduction](#introduction)
  - [Dataset Formatting](#dataset-formatting)
  - [Data Pre-Processing](#data-pre-processing)
  - [Training](#training)
  - [Vocoder Fine-Tuning](#vocoder-fine-tuning)
  - [TTS Synthesis](#tts-synthesis)
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

We recommend using conda for python environment management, for example download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  
Create your python environment and install dependencies using the Makefile:
1. `conda create -n daft_exprt python=3.8 -y`
2. `conda activate daft_exprt`
3. `cd environment`
4. `make`

All Linux/Conda/Pip dependencies will be installed by the Makefile, and the repository will be installed as a pip package in editable mode.

### Docker Image
Requirements:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx?lang=en-us) >= 450.80.02

Build the Docker image using the associated Dockerfile:  
1. `docker build -f environment/Dockerfile -t daft_exprt .`

## Quick Start Example

### Introduction
This quick start guide will illustrate how to use the different scripts of this repository to:
1. Format datasets
2. Pre-process these datasets
3. Train Daft-Exprt on the pre-processed data
4. Generate a dataset for vocoder fine-tuning
5. Use Daft-Exprt for TTS synthesis

All scripts are located in [scripts](scripts) directory.  
Daft-Exprt source code is located in [daft_exprt](src/daft_exprt) directory.  
Config parameters used in the scripts are all instanciated in [hparams.py](src/daft_exprt/hparams.py).  

As a quick start example, we consider using the 22kHz [LJ speech dataset](https://keithito.com/LJ-Speech-Dataset/) and the 16kHz emotional speech dataset (ESD) from [Zhou et al](https://github.com/HLTSingapore/Emotional-Speech-Data).  
This combines a total of 11 speakers. All speaker datasets must be in the same root directory. For example:
```
/data_dir
    LJ_Speech
    ESD
        spk_1
        ...
        spk_N
```

In this example, we use the docker image built in the previous section:
 ```
docker run -it --gpus all -v /path/to/data_dir:/workdir/data_dir -v path/to/repo_dir:/workdir/repo_dir IMAGE_ID
```


### Dataset Formatting
The source code expects the specific tree structure for each speaker data set:
```
/speaker_dir
    metadata.csv
    /wavs
        wav_file_name_1.wav
        ...
        wav_file_name_N.wav
```

metadata.csv must be formatted as follows:
```
wav_file_name_1|text_1
...
wav_file_name_N|text_N
```

Given each dataset has its own nomenclature, this project does not provide a ready-made universal script.  
However, the script [format_dataset.py](scripts/format_dataset.py) already proposes the code to format LJ and ESD:
```
python format_dataset.py \
    --data_set_dir /workdir/data_dir/LJ_Speech \
    LJ

python format_dataset.py \
    --data_set_dir /workdir/data_dir/ESD \
    ESD \
    --language english
```

### Data Pre-Processing
In this section, the code will:
1. Align data using MFA
2. Extract features for training
3. Create train and validation sets
4. Extract features stats on the train set for speaker standardization

To pre-process all available formatted data (i.e. LJ and ESD in this example):
```
python training.py \
    --experiment_name EXPERIMENT_NAME \
    --data_set_dir /workdir/data_dir \
    pre_process
```

This will pre-process data using the default hyper-parameters that are set for 22kHz audios.  
All outputs related to the experiment will be stored in `/workdir/repo_dir/trainings/EXPERIMENT_NAME`.  
You can also target specific speakers for data pre-processing. For example, to consider only ESD speakers:
```
python training.py \
    --experiment_name EXPERIMENT_NAME \
    --speakers ESD/spk_1 ... ESD/spk_N \
    --data_set_dir /workdir/data_dir \
    pre_process
```

The pre-process function takes several arguments:
- `--features_dir`: absolute path where pre-processed data will be stored. Default to `/workdir/repo_dir/datasets`
- `--proportion_validation`: Proportion of examples that will be in the validation set. Default to `0.1`% per speaker.
- `--nb_jobs`: number of cores to use for python multi-processing. If set to `max`, all CPU cores are used. Default to `6`.

Note that if it is the first time that you pre-process the data, this step will take several hours.  
You can decrease computing time by increasing the `--nb_jobs` parameter.  
    
### Training
Once pre-processing is finished, launch training. To train on all pre-processed data:
```
python training.py \
    --experiment_name EXPERIMENT_NAME \
    --data_set_dir /workdir/data_dir \
    train
```

Or if you targeted specific speakers during pre-processing (e.g. ESD speakers):
```
python training.py \
    --experiment_name EXPERIMENT_NAME \
    --speakers ESD/spk_1 ... ESD/spk_N \
    --data_set_dir /workdir/data_dir \
    train
```

All outputs related to the experiment will be stored in `/workdir/repo_dir/trainings/EXPERIMENT_NAME`.  

The train function takes several arguments:
- `--checkpoint`: absolute path of a Daft-Exprt checkpoint. Default to `""`
- `--no_multiprocessing_distributed`: disable PyTorch multi-processing distributed training. Default to `False`
- `--world_size`: number of nodes for distributed training. Default to `1`.
- `--rank`: node rank for distributed training. Default to `0`.
- `--master`: url used to set up distributed training. Default to `tcp://localhost:54321`.

These default values will launch a new training starting at iteration 0, using all available GPUs on the machine.  
The code supposes that only 1 GPU is available on the machine.  
Default [batch size](src/daft_exprt/hparams.py#L66) and [gradient accumulation](src/daft_exprt/hparams.py#L67) hyper-parameters are set to values to reproduce the batch size of 48 from the paper.

The code also supports tensorboard logging. To display logging outputs:  
`tensorboard --logdir_spec=EXPERIMENT_NAME:/workdir/repo_dir/trainings/EXPERIMENT_NAME/logs`

### Vocoder Fine-Tuning
Once training is finished, you can create a dataset for vocoder fine-tuning:
```
python training.py \
    --experiment_name EXPERIMENT_NAME \
    --data_set_dir /workdir/data_dir \
    fine_tune \
    --checkpoint CHECKPOINT_PATH
```

Or if you targeted specific speakers during pre-processing and training (e.g. ESD speakers):
```
python training.py \
    --experiment_name EXPERIMENT_NAME \
    --speakers ESD/spk_1 ... ESD/spk_N \
    --data_set_dir /workdir/data_dir \
    fine_tune \
    --checkpoint CHECKPOINT_PATH
```

Fine-tuning dataset will be stored in `/workdir/repo_dir/trainings/EXPERIMENT_NAME/fine_tuning_dataset`.  

### TTS Synthesis
For an example on how to use Daft-Exprt for TTS synthesis, run the script [synthesize.py](scripts/synthesize.py).  
```
python synthesize.py \
    --output_dir OUTPUT_DIR \
    --checkpoint CHECKPOINT
```

Default sentences and reference utterances are used in the script.  

The script also offers the possibility to:
- `--batch_size`: process batch of sentences in parallel
- `--real_time_factor`: estimate Daft-Exprt real time factor performance given the chosen batch size
- `--control`: perform local prosody control
- `--symbol_prosody_file`: bypass phonemization/prosody prediction by supplying tuples generated with `extract_symbol_prosody.py`
- `--use_griffin_lim`: keep the legacy Griffin-Lim reconstruction (otherwise a universal HiFi-GAN vocoder is used automatically)
- `--vocoder_checkpoint`: provide a custom HiFi-GAN generator checkpoint; if omitted, a pretrained universal 22kHz model is downloaded on demand

### Symbol-Level Prosody Extraction
Use [extract_symbol_prosody.py](scripts/extract_symbol_prosody.py) to align raw audio with its transcript and export per-symbol duration, pitch, and energy tuples that mirror the targets of the local prosody predictor. The script:
- reads a manifest where each line contains `absolute/path/to/audio.wav|verbatim transcript`
- looks up words in the provided MFA dictionary first, only running MFA G2P for out-of-vocabulary words (either `--g2p_preset american_english|indian_english` or `--g2p_model /path/to/model.zip`)
- runs Montreal Forced Aligner with the provided (or auto-generated) dictionary and MFA acoustic model
- extracts mel spectrograms, REAPER pitch, and converts frame-level values to phoneme-level tuples `(symbol, duration_in_frames, mean_pitch, mean_energy)`

Dependencies: MFA CLI (`mfa align` / `mfa g2p`), the matching acoustic/G2P packages (download with `mfa download acoustic ...` / `mfa download g2p ...`), and the `reaper` binary for F0 extraction. Provide an MFA dictionary via `--dictionary` or let the script reuse the default one bundled with the pretrained model.

Example:
```
python scripts/extract_symbol_prosody.py \
    --manifest /path/to/list.txt \
    --output /tmp/prosody.txt \
    --g2p_preset american_english \
    --nb_jobs 4 \
    --include_audio_path
```
Output file `/tmp/prosody.txt` will contain one line per audio. Each line is a Python-style list of tuples; optionally the original audio path is prefixed when `--include_audio_path` is set:
```
/abs/audio.wav|[('DH', 5, 3.512, 1.104), ('AH0', 4, 3.403, 1.089), ('S', 3, 0.000, 0.982), ...]
```
Pass `--mix_output False` if you instead need four parallel arrays (symbols, durations, pitch, energy):
```
/abs/audio.wav|(['DH', 'AH0', 'S', ...], [5, 4, 3, ...], [3.512, 3.403, 0.0, ...], [1.104, 1.089, 0.982, ...])
```
These tuples can replace the local prosody predictor outputs when performing deterministic prosody control.

## Evaluation Utilities

### WER and Accent Confidence Tracking
To benchmark accent conversion quality we provide `scripts/evaluation/compute_wer_and_accent_metrics.py`.  
It ingests two manifests (pre- and post-conversion) formatted as `utt_id|/abs/or/rel/path.wav|Reference transcript` (or without the `utt_id`, where the filename stem is used), transcribes the converted audio with an OpenAI Whisper checkpoint, and scores both audio versions with the SpeechBrain `Jzuluaga/accent-id-commonaccent_xlsr-en-english` custom interface. The script emits a JSON report that lists per-utterance WER, transcripts, and source/target accent confidences (with deltas), plus global micro WER and average confidence shifts. If `--plot-path` is provided an overlapping histogram of the predicted accent class distributions before vs. after conversion is also saved.

```
python scripts/evaluation/compute_wer_and_accent_metrics.py \
    --before-manifest /path/to/source_manifest.txt \
    --after-manifest /path/to/converted_manifest.txt \
    --source-class indian \
    --target-class american \
    --output-json outputs/indian_to_american_metrics.json \
    --plot-path outputs/indian_to_american_hist.png
```

Key options let you pick which Whisper size to run (`--whisper-model`), choose where the SpeechBrain cache lives (`--accent-cache-dir`), change the manifest separator, disable accent prediction caching (`--disable-accent-cache`), set an artifact prefix (`--experiment-name`), or cap the number of processed utterances (useful for smoke tests). Relative audio paths are resolved against their manifest location, and both manifests must expose the same utterance identifiers in the same order. Source/target accents should be one of the CommonAccent labels used by the classifier (`united_states`, `england`, `australia`, `india`, `canada`, `bermuda`, `scotland`, `africa`, `ireland`, `new_zealand`, `wales`, `malaysia`, `philippines`, `singapore`, `hong_kong`, `south_atlantic`), with common synonyms like `american` automatically mapped.

## Citation
```
@article{Zaidi2021,
abstract = {},
journal = {arXiv},
arxivId = {2108.02271},
author = {Za{\"{i}}di, Julian and Seut{\'{e}}, Hugo and van Niekerk, Benjamin and Carbonneau, Marc-Andr{\'{e}}},
eprint = {2108.02271},
title = {{Daft-Exprt: Robust Prosody Transfer Across Speakers for Expressive Speech Synthesis}},
url = {https://arxiv.org/pdf/2108.02271.pdf},
year = {2021}
}
```

## Contributing
Any contribution to this repository is more than welcome!  
If you have any feedback, please send it to julian.zaidi@ubisoft.com.  


© [2021] Ubisoft Entertainment. All Rights Reserved
