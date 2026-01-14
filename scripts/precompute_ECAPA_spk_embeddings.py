#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import torch
import numpy as np
import torchaudio
try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def precompute_embeddings(hparams, output_dir_base):
    # Load SpeechBrain ECAPA-TDNN model
    # We use a pre-trained model from HuggingFace
    logger.info("Loading SpeechBrain ECAPA-TDNN model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    
    # Process training and validation files
    file_lists = [hparams['training_files'], hparams['validation_files']]
    
    # We need to resolve file paths. The training files usually contain:
    # /path/to/wavs/filename|transcript...
    # But wait, looking at data_loader, it processes lines differently. 
    # Let's see DaftExprtDataLoader logic: 
    # data[0] is features_dir, data[1] is feature_file.
    # The training file line format is: features_dir|feature_file|speaker_id|...
    
    count = 0
    for file_list in file_lists:
        if not os.path.exists(file_list):
            logger.warning(f"File list {file_list} not found, skipping.")
            continue
            
        with open(file_list, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('|') for line in f]
            
        for line in lines:
            features_dir = line[0]
            feature_file = line[1]
            speaker_id = line[2] # Unused here but good to know
            
            # The wav file location is usually inferred or we need to find it. 
            # In Daft-Exprt preprocessing, the wavs are often in 'wavs' dir parallel to features 'align' or similar.
            # However, `extract_features.py` puts features in `dataset_dir/speaker/features`.
            
            # Let's try to locate the original wav. 
            # If features_dir is `.../speaker/features`, wavs might me in `.../speaker/wavs/{feature_file}.wav`
            # Or we can just use the audio loading logic if we can find the path.
            # BUT, we might not have the raw wav path in the file list directly.
            
            # Alternative: Re-read the metadata used for extraction if possible.
            # Or assume a standard directory structure.
            # Standard structure: ${Dataset}/${Speaker}/wavs/${File}.wav
            
            # Only reliable way is if we know the root structure.
            # Let's assume features_dir identifies the speaker folder.
            # features_dir usually ends with .../features or just is the folder.
            
            # Let's try to construct wav path.
            # If features_dir is `.../features/{speaker}`, then wav is `.../wavs/{file}.wav`?
            # Or `.../{speaker}/wavs/{file}.wav`.
            
            # Let's try to detect based on file existence.
            # Common pattern in this repo: dataset/speaker/wavs/file.wav
            
            # If features_dir is absolute, we can try to backtrack.
            # But wait, we can just look for the wav file relative to features_dir.
            
            # Try multiple relative paths relative to features_dir
            # 1. Standard: ../wavs/file.wav (if features in speaker dir)
            # 2. Parallel: ../../../tiny_datasets/{Using Dataset Name?}/wavs
            # 3. Simple assumption: If feature_dir is .../X, Look for wavs in .../X/wavs 
            # OR Look for wavs in .../../wavs
            
            # Let's try a few known patterns for this codebase
            candidates = [
                os.path.join(features_dir, "../wavs", f"{feature_file}.wav"),
                os.path.join(features_dir, "wavs", f"{feature_file}.wav"),
                os.path.join(features_dir, f"{feature_file}.wav"),
                # Handle tiny_datasets vs features_tiny structure
                # features_dir: .../datasets/features_tiny/LJSpeech
                # wav: .../datasets/tiny_datasets/LJSpeech/wavs/LJ001-0001.wav
                os.path.join(features_dir, "../../tiny_datasets", os.path.basename(features_dir), "wavs", f"{feature_file}.wav")
            ]
            
            found_wav = None
            for cand in candidates:
                if os.path.exists(cand):
                    found_wav = cand
                    break
                    
            if found_wav is None:
                # Last ditch: walk? No too slow.
                logger.warning(f"Could not find wav file for {feature_file} (tried {candidates[0]} etc...), skipping.")
                continue
                
            # Check if embedding already exists
            emb_path = os.path.join(features_dir, f"{feature_file}.spk_emb.npy")
            if os.path.exists(emb_path):
                continue
                
            # Load Audio
            signal, fs = torchaudio.load(found_wav)
            # ECAPA expects 16kHz usually, let's check classifier expectations or just pass it.
            # SpeechBrain usually handles resampling or expects specific SR. 
            # ecapa-voxceleb is 16k.
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                signal = resampler(signal)
            
            # Compute Embedding
            # signal should be (batch, time)
            # torchaudio loads as (channel, time). If mono, (1, time).
            embeddings = classifier.encode_batch(signal)
            # embeddings shape: (batch, 1, 192) -> squeeze to (192,)
            emb_vector = embeddings.squeeze().cpu().numpy()
            
            # Save
            np.save(emb_path, emb_vector)
            
            count += 1
            if count % 100 == 0:
                logger.info(f"Processed {count} files...")

    logger.info(f"Finished. Processed {count} new embeddings.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.json used for training (to get file lists)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        hparams = json.load(f)
        
    precompute_embeddings(hparams, None)
