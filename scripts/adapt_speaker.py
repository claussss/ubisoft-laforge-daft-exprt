import argparse
import json
import logging
import os
import shutil
import time
import random
import numpy as np
import torch
from torch.optim import Adam
from daft_exprt.hparams import HyperParams
from daft_exprt.model import DaftExprt, get_mask_from_lengths
from daft_exprt.loss import DaftExprtLoss
from daft_exprt.data_loader import DaftExprtDataLoader, DaftExprtDataCollate
from torch.utils.data import DataLoader
from daft_exprt.extract_features import _extract_features
from daft_exprt.mfa import prepare_corpus, extract_markers, move_file
from daft_exprt.utils import launch_multi_process

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

def setup_logger(log_file):
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ],
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )

def run_preprocessing(adapt_dir, hparams, n_jobs=1):
    """
    Run MFA alignment and Feature Extraction if needed.
    Expects:
        adapt_dir/wavs/*.wav
        adapt_dir/metadata.csv (file_name|text)
    """
    _logger.info("Checking preprocessing status...")
    
    speaker_name = "adaptation_speaker"
    # Create structure if not exists
    # We treat adapt_dir as the "corpus_dir" for this speaker
    # Structure expected by MFA/ExtractFeatures:
    # adapt_dir/wavs
    # adapt_dir/align (output of MFA)
    # adapt_dir/speaker_adapt (output of extract_features)
    
    wavs_dir = os.path.join(adapt_dir, 'wavs')
    align_dir = os.path.join(adapt_dir, 'align')
    features_dir = os.path.join(adapt_dir, 'speaker_adapt')
    
    if not os.path.isdir(wavs_dir):
        _logger.error(f"Wavs directory missing: {wavs_dir}")
        raise FileNotFoundError(f"Wavs directory missing: {wavs_dir}")

    # 1. MFA Alignment
    if not os.path.isdir(align_dir) or len(os.listdir(align_dir)) == 0:
        _logger.info("Running MFA Alignment...")
        os.makedirs(align_dir, exist_ok=True)
        
        # Prepare corpus (create .lab files)
        prepare_corpus(adapt_dir, hparams.language)
        
        # Run MFA
        dictionary = hparams.mfa_dictionary
        acoustic_model = hparams.mfa_acoustic_model
        temp_dir = os.path.join(adapt_dir, 'tmp_mfa')
        
        cmd = f'mfa align {adapt_dir} {dictionary} {acoustic_model} {align_dir} -t {temp_dir} -j {n_jobs} -v -c'
        _logger.info(f"Executing: {cmd}")
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError("MFA alignment failed.")
            
        # Cleanup MFA output (move TextGrids/wavs if needed, extract markers)
        # MFA output structure in align_dir depends on version, but usually has .TextGrid
        # mfa.py logic:
        text_grid_dir = os.path.join(align_dir, 'wavs') # MFA often outputs here
        if os.path.isdir(text_grid_dir):
             all_files = [x for x in os.listdir(text_grid_dir)]
             launch_multi_process(iterable=all_files, func=move_file, n_jobs=n_jobs,
                                  src_dir=text_grid_dir, dst_dir=align_dir, timer_verbose=False)
             shutil.rmtree(text_grid_dir, ignore_errors=True)
        
        extract_markers(align_dir, n_jobs)
        
        # Move .lab files to align dir
        lab_files = [x for x in os.listdir(wavs_dir) if x.endswith('.lab')]
        launch_multi_process(iterable=lab_files, func=move_file, n_jobs=n_jobs,
                             src_dir=wavs_dir, dst_dir=align_dir, timer_verbose=False)
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        _logger.info("Alignment directory exists, skipping MFA.")

    # 2. Feature Extraction
    if not os.path.isdir(features_dir) or len(os.listdir(features_dir)) == 0:
        _logger.info("Running Feature Extraction...")
        os.makedirs(features_dir, exist_ok=True)
        
        # Identify files to process
        # We look for .markers in align_dir
        markers = [x for x in os.listdir(align_dir) if x.endswith('.markers')]
        files_to_process = []
        for m in markers:
            basename = m.replace('.markers', '')
            wav_path = os.path.join(wavs_dir, f"{basename}.wav")
            marker_path = os.path.join(align_dir, m)
            if os.path.exists(wav_path):
                files_to_process.append((marker_path, wav_path))
        
        launch_multi_process(iterable=files_to_process, func=_extract_features, n_jobs=n_jobs,
                             features_dir=features_dir, hparams=hparams)
    else:
        _logger.info("Features directory exists, skipping extraction.")
        
    return features_dir

def compute_speaker_stats(features_dir):
    """
    Compute Mean/Std for Pitch and Energy from extracted features.
    """
    _logger.info("Computing Speaker Stats...")
    pitch_files = [os.path.join(features_dir, x) for x in os.listdir(features_dir) if x.endswith('.frames_f0')]
    energy_files = [os.path.join(features_dir, x) for x in os.listdir(features_dir) if x.endswith('.frames_nrg')]
    
    all_pitch = []
    for f in pitch_files:
        with open(f, 'r') as pf:
            vals = [float(line.strip()) for line in pf]
            # Filter unvoiced (0.0) and log-scale padding (if any)
            # DaftExprt uses log pitch, 0.0 for unvoiced.
            vals = [v for v in vals if v > 0.0]
            all_pitch.extend(vals)
            
    all_energy = []
    for f in energy_files:
        with open(f, 'r') as ef:
            vals = [float(line.strip()) for line in ef]
            # Filter 0 energy (padding)
            vals = [v for v in vals if v > 0.0]
            all_energy.extend(vals)
            
    stats = {
        'pitch': {
            'mean': float(np.mean(all_pitch)) if all_pitch else 0.0,
            'std': float(np.std(all_pitch)) if all_pitch else 1.0
        },
        'energy': {
            'mean': float(np.mean(all_energy)) if all_energy else 0.0,
            'std': float(np.std(all_energy)) if all_energy else 1.0
        }
    }
    _logger.info(f"Computed Stats: {stats}")
    return stats

def smart_initialization(model, adapt_loader, device, n_existing_speakers):
    """
    Find the best existing speaker embedding to initialize the new speaker.
    Minimizes Mel L1 Loss on the adaptation batch.
    """
    _logger.info("Running Smart Initialization...")
    model.eval()
    best_loss = float('inf')
    best_spk_id = 0
    
    # We only need one batch (assuming full batch adaptation)
    batch = next(iter(adapt_loader))
    inputs, targets = model.parse_batch(device, batch)
    # inputs: (..., speaker_ids)
    # We need to override speaker_ids in inputs to test each speaker
    
    # Unpack inputs to modify speaker_ids
    # inputs tuple: (symbols, dur_float, dur_int, sym_nrg, sym_pitch, in_lens, fr_nrg, fr_pitch, mel, out_lens, spk_ids)
    input_list = list(inputs)
    
    with torch.no_grad():
        for spk_id in range(n_existing_speakers):
            # Create dummy speaker tensor
            batch_size = input_list[0].size(0)
            test_spk_ids = torch.full((batch_size,), spk_id, dtype=torch.long, device=device)
            input_list[-1] = test_spk_ids
            test_inputs = tuple(input_list)
            
            # Forward
            outputs = model(test_inputs)
            # outputs: (mel_preds, ...)
            mel_preds = outputs[0]
            mel_targets = targets[3] # targets is tuple (dur, nrg, pitch, mel, len, pitch_fr)
            output_lengths = targets[4]
            
            # Compute L1 Loss
            loss = torch.nn.L1Loss(reduction='none')(mel_preds, mel_targets)
            # Masking
            mask = ~get_mask_from_lengths(output_lengths)
            loss = loss.masked_fill(mask.unsqueeze(1), 0)
            loss = torch.sum(loss) / torch.sum(output_lengths)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_spk_id = spk_id
                
    _logger.info(f"Best initialization found: Speaker {best_spk_id} (Loss: {best_loss:.4f})")
    
    # Copy weights
    with torch.no_grad():
        model.spk_embedding.weight[n_existing_speakers] = model.spk_embedding.weight[best_spk_id]

def adapt_speaker(args):
    # 1. Load Config
    with open(args.config_file) as f:
        data = f.read()
    config = json.loads(data)
    hparams = HyperParams(verbose=False, **config)
    
    # 2. Preprocessing
    features_dir = run_preprocessing(args.adapt_dir, hparams)
    
    # 3. Stats
    if args.speaker_stats:
        with open(args.speaker_stats, 'r') as f:
            new_stats = json.load(f)
    else:
        new_stats = compute_speaker_stats(features_dir)
    
    # Update hparams with new speaker
    new_spk_name = args.output_name
    # We assign a temporary ID for the new speaker (N)
    # But hparams.speakers is a list of names.
    # We need to ensure the model knows about N+1 speakers.
    # The stats dict uses keys 'spk ID'.
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Checkpoint
    _logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict']
    
    # Handle DataParallel prefix 'module.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    # Check n_speakers in checkpoint
    if 'spk_embedding.weight' in state_dict:
        ckpt_n_speakers = state_dict['spk_embedding.weight'].size(0)
        if ckpt_n_speakers != hparams.n_speakers:
            _logger.warning(f"Config n_speakers ({hparams.n_speakers}) != Checkpoint n_speakers ({ckpt_n_speakers}). Updating config.")
            hparams.n_speakers = ckpt_n_speakers
    else:
        raise KeyError("Could not find 'spk_embedding.weight' in checkpoint state_dict")

    # Initialize model with N+1 speakers (for the new one)
    hparams.n_speakers += 1
    model = DaftExprt(hparams).to(device)
    
    # Handle Embedding Resize in State Dict
    # The checkpoint has N speakers. Model has N+1.
    old_emb_weight = state_dict['spk_embedding.weight'] # (N, dim)
    new_emb_weight = model.spk_embedding.weight # (N+1, dim)
    
    # Copy old weights
    with torch.no_grad():
        new_emb_weight[:old_emb_weight.size(0)] = old_emb_weight
        
    # Remove embedding from state_dict to avoid size mismatch error during load
    del state_dict['spk_embedding.weight']
    model.load_state_dict(state_dict, strict=False)
    
    # Restore embedding weights (now containing old + random new)
    # Actually we already set them in the model parameter, so we are good.
    
    # 5. Data Loader
    # Generate a temporary file list
    # Format: features_dir|feature_file|speaker_id
    # feature_file is filename without extension
    files = [x.replace('.npy', '') for x in os.listdir(features_dir) if x.endswith('.npy')]
    _logger.info(f"Found {len(files)} samples for adaptation.")
    new_spk_id = hparams.n_speakers - 1
    _logger.info(f"New Speaker ID: {new_spk_id}")
    
    # Update stats in hparams for the new ID
    # This is crucial for DataLoader normalization
    hparams.stats[f'spk {new_spk_id}'] = new_stats
    _logger.info(f"Added stats for 'spk {new_spk_id}'. Current stats keys: {list(hparams.stats.keys())}")
    
    # Update speakers list to match n_speakers
    if len(hparams.speakers) < hparams.n_speakers:
        hparams.speakers.append(f"{args.output_name}")
        hparams.speakers_id.append(new_spk_id)
        _logger.info(f"Updated hparams.speakers: {hparams.speakers}")
    
    temp_list_file = os.path.join(args.adapt_dir, 'adapt_file_list.txt')
    with open(temp_list_file, 'w') as f:
        for fname in files:
            f.write(f"{features_dir}|{fname}|{new_spk_id}\n")
            
    # Create Dataset
    # We mock hparams.training_files
    hparams.training_files = temp_list_file
    dataset = DaftExprtDataLoader(temp_list_file, hparams, shuffle=True)
    collate_fn = DaftExprtDataCollate(hparams)
    loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn, shuffle=True)
    
    # 6. Smart Initialization
    smart_initialization(model, loader, device, new_spk_id)
    
    # 7. Freeze/Unfreeze (Tiers)
    for param in model.parameters():
        param.requires_grad = False
        
    # Tier 1: Embedding
    model.spk_embedding.weight.requires_grad = True
    # We only want to train the NEW embedding vector, but PyTorch embedding layer gradients are sparse.
    # We can just zero out gradients for other indices or trust the optimizer since we only feed new_spk_id.
    
    if args.tier >= 2:
        # Tier 2: + Bias
        model.frame_decoder.projection.linear_layer.bias.requires_grad = True
        
    if args.tier >= 3:
        # Tier 3: + FiLM
        model.gammas_predictor.linear_layer.bias.requires_grad = True
        model.betas_predictor.linear_layer.bias.requires_grad = True
        
    # Optimizer
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Disable consistency losses for adaptation (avoids loading pitch predictor if missing)
    hparams.pitch_consistency_weight = 0.0
    hparams.energy_consistency_weight = 0.0
    hparams.adversarial_weight = 0.0
    
    criterion = DaftExprtLoss(device, hparams)
    
    # Compute Target Norm
    target_norm = None
    if args.manifold_constraint:
        with torch.no_grad():
            # Exclude the new speaker (last one)
            existing_embs = model.spk_embedding.weight[:-1]
            norms = torch.norm(existing_embs, dim=1)
            target_norm = torch.mean(norms).item()
            _logger.info(f"Manifold Constraint Enabled. Target Norm: {target_norm:.4f}")

    # 8. Training Loop
    _logger.info(f"Starting Adaptation (Tier {args.tier})...")
    model.train()
    
    for i in range(args.iterations): # Iterations
        for batch in loader:
            optimizer.zero_grad()
            inputs, targets = model.parse_batch(device, batch)
            outputs = model(inputs)
            
            # Loss
            criterion.adversarial_weight = 0.0
            criterion.energy_consistency_weight = 0.0
            criterion.pitch_consistency_weight = 0.0
            
            loss, _ = criterion(outputs, targets, iteration=i)
            
            loss.backward()
            
            # Zero out gradients for old speakers
            if model.spk_embedding.weight.grad is not None:
                mask = torch.ones_like(model.spk_embedding.weight.grad)
                mask[new_spk_id] = 0
                model.spk_embedding.weight.grad.masked_fill_(mask.bool(), 0.0)
                
            optimizer.step()
            
            # Apply Manifold Constraint
            if args.manifold_constraint and target_norm is not None:
                with torch.no_grad():
                    current_emb = model.spk_embedding.weight[new_spk_id]
                    current_norm = torch.norm(current_emb)
                    new_emb = current_emb * (target_norm / current_norm)
                    model.spk_embedding.weight[new_spk_id] = new_emb
            
        if i % 20 == 0:
            _logger.info(f"Iter {i}: Loss {loss.item():.4f}")
            
    # 9. Save Checkpoint
    output_path = os.path.join(args.adapt_dir, f"{args.output_name}.pt")
    _logger.info(f"Saving adapted model to {output_path}")
    
    # We need to save the new n_speakers and stats in the config
    # The checkpoint dict contains 'config_params'
    new_config = hparams.__dict__.copy()
    
    final_state_dict = model.state_dict()
    # If adversarial weight is 0, strip adversary keys to prevent loading errors
    if hparams.adversarial_weight == 0:
        keys_to_remove = [k for k in final_state_dict.keys() if 'adversary' in k]
        if keys_to_remove:
            _logger.info(f"Stripping {len(keys_to_remove)} unused adversary keys from checkpoint.")
            for k in keys_to_remove:
                del final_state_dict[k]
    
    torch.save({
        'iteration': 0,
        'state_dict': final_state_dict,
        'optimizer': optimizer.state_dict(),
        'config_params': new_config
    }, output_path)
    
    # Clean up
    if os.path.exists(temp_list_file):
        os.remove(temp_list_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--adapt_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--tier', type=int, default=2, choices=[1, 2, 3])
    parser.add_argument('--speaker_stats', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--manifold_constraint', action='store_true', default=False, help="Constrain embedding norm to mean of existing speakers")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay for optimizer")
    
    args = parser.parse_args()
    set_seed(1234)
    setup_logger(os.path.join(args.adapt_dir, 'adapt.log'))
    adapt_speaker(args)
