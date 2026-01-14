import torch
import numpy as np
import sys

ckpt_path = "DaftExprt_65000_adv_more_consist.pt"
adapt_path = "spk_adapt_4_constrained.pt"

print(f"Analyzing Embedding Manifold (Constrained)...")

try:
    # Load Original
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['state_dict']
    
    if 'module.spk_embedding.weight' in state:
        embs = state['module.spk_embedding.weight']
    elif 'spk_embedding.weight' in state:
        embs = state['spk_embedding.weight']
    else:
        print("Error: Could not find embeddings in original checkpoint")
        sys.exit(1)
        
    # embs is (N_speakers, Dim)
    # Exclude the last one if it was a previous adaptation or unused
    # Assuming first 11 (0-10) are the valid pre-trained speakers
    valid_embs = embs[:11] 
    
    norms = torch.norm(valid_embs, dim=1)
    mean_norm = torch.mean(norms).item()
    std_norm = torch.std(norms).item()
    
    print(f"\nOriginal Speakers (0-10):")
    print(f"  Norm Mean: {mean_norm:.4f}")
    print(f"  Norm Std:  {std_norm:.4f}")
    print(f"  Min Norm:  {torch.min(norms).item():.4f}")
    print(f"  Max Norm:  {torch.max(norms).item():.4f}")
    
    # Calculate pairwise distances to see "density"
    dists = []
    for i in range(len(valid_embs)):
        for j in range(i+1, len(valid_embs)):
            d = torch.norm(valid_embs[i] - valid_embs[j]).item()
            dists.append(d)
    
    mean_dist = np.mean(dists)
    print(f"  Avg Pairwise Dist: {mean_dist:.4f}")
    
    # Load Adapted
    adapt_ckpt = torch.load(adapt_path, map_location='cpu')
    adapt_state = adapt_ckpt['state_dict']
    
    # Adapted ID is 12
    if 'spk_embedding.weight' in adapt_state:
        adapt_emb = adapt_state['spk_embedding.weight'][12]
    elif 'module.spk_embedding.weight' in adapt_state:
        adapt_emb = adapt_state['module.spk_embedding.weight'][12]
    else:
        print("Error: Could not find adapted embedding")
        sys.exit(1)
        
    adapt_norm = torch.norm(adapt_emb).item()
    print(f"\nAdapted Speaker (Aggressive):")
    print(f"  Norm: {adapt_norm:.4f}")
    
    # Distance to Mean Embedding
    mean_emb = torch.mean(valid_embs, dim=0)
    dist_to_mean = torch.norm(adapt_emb - mean_emb).item()
    print(f"  Dist to Mean Emb: {dist_to_mean:.4f}")
    
    # Distance to Nearest Original
    dists_to_orig = [torch.norm(adapt_emb - e).item() for e in valid_embs]
    min_dist = min(dists_to_orig)
    print(f"  Dist to Nearest Orig: {min_dist:.4f}")
    
    # Z-Score of Norm
    z_score = (adapt_norm - mean_norm) / std_norm
    print(f"  Norm Z-Score: {z_score:.2f}")
    
    if abs(z_score) > 2.0:
        print("\n[CONCLUSION] The adapted embedding has an abnormal norm (Out of Distribution).")
    elif min_dist > mean_dist * 1.5:
        print("\n[CONCLUSION] The adapted embedding is very far from any known speaker (Out of Distribution).")
    else:
        print("\n[CONCLUSION] The adapted embedding seems to be within the valid manifold statistics.")

except Exception as e:
    print(f"Error: {e}")
