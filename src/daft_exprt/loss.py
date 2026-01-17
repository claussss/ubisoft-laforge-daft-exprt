import torch

from torch import nn


from daft_exprt.layers.pitch_predictor import PitchPredictor

class DaftExprtLoss(nn.Module):
    def __init__(self, device, hparams):
        super(DaftExprtLoss, self).__init__()
        self.nb_channels = hparams.n_mel_channels
        self.mel_spec_weight = hparams.mel_spec_weight
        
        self.L1Loss = nn.L1Loss(reduction='none').to(device)
        self.MSELoss = nn.MSELoss(reduction='none').to(device)
        
        # Weights
        self.adversarial_weight = getattr(hparams, 'adversarial_weight', 0.0)
        self.energy_consistency_weight = getattr(hparams, 'energy_consistency_weight', 0.0)
        
        # Accent adversarial loss config (NEW)
        self.warmup_steps = getattr(hparams, 'warmup_steps', 10000)
        self.adv_max_weight = getattr(hparams, 'adv_max_weight', 1e-2)
        
        # Smoothing for consistency loss
        self.avg_pool = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        
        # Pitch Predictor for consistency loss
        self.pitch_predictor = None
        self.pitch_consistency_weight = getattr(hparams, 'pitch_consistency_weight', 0.0)
        if self.pitch_consistency_weight > 0 and hasattr(hparams, 'pitch_predictor_path'):
            self.pitch_predictor = PitchPredictor(n_mel_channels=hparams.n_mel_channels)
            # Load checkpoint
            checkpoint = torch.load(hparams.pitch_predictor_path, map_location=device)
            self.pitch_predictor.load_state_dict(checkpoint)
            self.pitch_predictor.to(device)
            self.pitch_predictor.eval()
            # Freeze
            for param in self.pitch_predictor.parameters():
                param.requires_grad = False
    
    def forward(self, outputs, targets, iteration):
        ''' Compute training loss

        :param outputs:         outputs predicted by the model
        :param targets:         ground-truth targets
        :param iteration:       current training iteration
        '''
        # extract ground-truth targets
        # targets can be 6 or 7 elements (backward compatibility)
        # targets = (durations_float, symbols_energy, symbols_pitch, mel_specs, output_lengths, frames_pitch, [speaker_ids])
        if len(targets) == 7:
            _, symbols_energy, symbols_pitch, mel_spec_targets, output_lengths, frames_pitch, speaker_ids = targets
        else:
            _, symbols_energy, symbols_pitch, mel_spec_targets, output_lengths, frames_pitch = targets
            speaker_ids = None
        mel_spec_targets.requires_grad = False
        
        # extract predictions
        # outputs = (mel_preds, durations, pitch_preds, energy_preds, src_mask, mel_mask, src_lens, mel_lens, attn_logprobs, adversary_preds, spk_preds)
        mel_spec_preds = outputs[0]
        input_lengths = outputs[6]
        adversary_preds = outputs[9] if len(outputs) > 9 else None
        spk_preds = outputs[10] if len(outputs) > 10 else None
            
        # compute mel-spec loss
        mel_spec_l1_loss = self.L1Loss(mel_spec_preds, mel_spec_targets)  # (B, n_mel_channels, T_max)
        mel_spec_l2_loss = self.MSELoss(mel_spec_preds, mel_spec_targets)  # (B, n_mel_channels, T_max)
        # divide by length of each sequence in the batch
        mel_spec_l1_loss = torch.sum(mel_spec_l1_loss, dim=(1, 2)) / (self.nb_channels * output_lengths)  # (B, )
        mel_spec_l1_loss = torch.mean(mel_spec_l1_loss)
        mel_spec_l2_loss = torch.sum(mel_spec_l2_loss, dim=(1, 2)) / (self.nb_channels * output_lengths)  # (B, )
        mel_spec_l2_loss = torch.mean(mel_spec_l2_loss)

        # add weights
        mel_spec_l1_loss = self.mel_spec_weight * mel_spec_l1_loss
        mel_spec_l2_loss = self.mel_spec_weight * mel_spec_l2_loss
        
        loss = mel_spec_l1_loss + mel_spec_l2_loss
        
        # Adversarial Loss
        adv_loss = torch.tensor(0.0, device=loss.device)
        if self.adversarial_weight > 0 and adversary_preds is not None:
            # adversary_preds: (B, T_src, 2)
            # targets: symbols_pitch (B, T_src), symbols_energy (B, T_src)
            # Stack targets: (B, T_src, 2)
            adv_targets = torch.stack([symbols_pitch, symbols_energy], dim=-1)
            
            # Mask out padding (using src_mask if available, or just ignore 0s if normalized?)
            # Ideally we use src_lens. But we don't have them here easily without passing more data.
            # Let's assume padded values are 0 and we want to ignore them or they are handled.
            # Actually, symbols_pitch/energy are 0-padded.
            # We should mask the loss.
            # Let's use a simple mask based on non-zero energy/pitch or just trust the model to learn 0s for padding.
            # Better: use the fact that we have output_lengths (mel lengths), but we need input_lengths for symbols.
            # We don't have input_lengths here. 
            # Let's assume the adversary learns to predict 0 for padding.
            
            adv_mse = self.MSELoss(adversary_preds, adv_targets) # (B, T_src, 2)
            
            # Masking
            # Create mask from input_lengths
            mask = torch.arange(adv_mse.size(1), device=adv_mse.device).expand(len(input_lengths), adv_mse.size(1)) < input_lengths.unsqueeze(1)
            # Expand mask for the 2 channels (pitch, energy)
            mask = mask.unsqueeze(-1).expand_as(adv_mse)
            
            adv_mse = adv_mse * mask.float()
            
            # Normalize by number of valid symbols
            adv_loss = torch.sum(adv_mse) / (torch.sum(mask.float()) + 1e-5)
            loss += self.adversarial_weight * adv_loss

        # Energy Consistency Loss
        energy_consistency_loss = torch.tensor(0.0, device=loss.device)
        if self.energy_consistency_weight > 0:
            # Calculate energy from predicted mel-spec
            # mel_spec_preds: (B, n_mel, T)
            # Energy = L2 norm of magnitudes (exp(mel))
            pred_magnitudes = torch.exp(mel_spec_preds)
            pred_energy = torch.norm(pred_magnitudes, dim=1) # (B, T)
            
            # Calculate energy from target mel-spec (to be fair comparison)
            target_magnitudes = torch.exp(mel_spec_targets)
            target_energy = torch.norm(target_magnitudes, dim=1) # (B, T)
            
            # Smooth both
            # Unsqueeze for AvgPool1d: (B, 1, T)
            pred_energy_smooth = self.avg_pool(pred_energy.unsqueeze(1)).squeeze(1)
            target_energy_smooth = self.avg_pool(target_energy.unsqueeze(1)).squeeze(1)
            
            # MSE
            consistency_mse = self.MSELoss(pred_energy_smooth, target_energy_smooth) # (B, T)
            # Mask by output_lengths?
            # We have output_lengths.
            # Create mask
            mask = torch.arange(consistency_mse.size(1), device=consistency_mse.device).expand(len(output_lengths), consistency_mse.size(1)) < output_lengths.unsqueeze(1)
            consistency_mse = consistency_mse * mask.float()
            
            energy_consistency_loss = torch.sum(consistency_mse) / torch.sum(output_lengths)
            loss += self.energy_consistency_weight * energy_consistency_loss

        # Pitch Consistency Loss
        pitch_consistency_loss = torch.tensor(0.0, device=loss.device)
        if self.pitch_predictor is not None:
            # Predict pitch from predicted mel-spec
            # mel_spec_preds: (B, n_mel, T)
            pred_pitch = self.pitch_predictor(mel_spec_preds) # (B, T)
            
            # Target pitch is frames_pitch (B, T)
            # We need to extract it from targets or inputs.
            # targets tuple currently: (durations_float, symbols_energy, symbols_pitch, mel_specs, output_lengths)
            # It does NOT contain frames_pitch.
            # We need to update model.py to pass frames_pitch in targets.
            # For now, let's assume we can't calculate it without frames_pitch.
            # Wait, we can use the pitch predictor on the TARGET mel-specs as a proxy for ground truth pitch if we trust it,
            # OR we must pass ground truth frames_pitch.
            # Passing ground truth frames_pitch is better.
            # Let's assume we will update model.py to include frames_pitch in targets.
            # New targets structure: (durations_float, symbols_energy, symbols_pitch, mel_specs, output_lengths, frames_pitch)
            
            # Placeholder: if len(targets) > 5: frames_pitch = targets[5]
            if len(targets) > 5:
                frames_pitch = targets[5]
                frames_pitch.requires_grad = False
                
                # MSE
                pitch_mse = self.MSELoss(pred_pitch, frames_pitch) # (B, T)
                
                # Mask
                # 1. Length masking
                len_mask = torch.arange(pitch_mse.size(1), device=pitch_mse.device).expand(len(output_lengths), pitch_mse.size(1)) < output_lengths.unsqueeze(1)
                # 2. Unvoiced masking (ignore frames where GT pitch is 0)
                # frames_pitch is 0.0 for unvoiced frames (explicitly set in data_loader)
                unvoiced_mask = (frames_pitch != 0.0)
                
                combined_mask = len_mask & unvoiced_mask
                
                pitch_mse = pitch_mse * combined_mask.float()
                
                # Normalize by number of voiced frames (plus epsilon)
                pitch_consistency_loss = torch.sum(pitch_mse) / (torch.sum(combined_mask.float()) + 1e-5)
                loss += self.pitch_consistency_weight * pitch_consistency_loss
        
        # Accent Adversarial Loss (NEW)
        accent_adv_loss = torch.tensor(0.0, device=loss.device)
        
        if spk_preds is not None and speaker_ids is not None:
            # Apply warmup for adversarial weight
            warmup_steps = getattr(self, 'warmup_steps', 10000)
            adv_max_weight = getattr(self, 'adv_max_weight', 1e-2)
            
            # Gradual increase: weight = min(adv_max_weight, iteration * warmup^-1.5 * adv_max / warmup^-0.5)
            weight_iter = iteration * (warmup_steps ** -1.5) * adv_max_weight / (warmup_steps ** -0.5)
            weight = min(adv_max_weight, weight_iter)
            
            # Cross-entropy loss for speaker classification
            accent_adv_loss = weight * torch.nn.functional.cross_entropy(spk_preds, speaker_ids)
            loss += accent_adv_loss

        # create individual loss tracker
        individual_loss = {
            'mel_spec_l1_loss': mel_spec_l1_loss.item(), 
            'mel_spec_l2_loss': mel_spec_l2_loss.item(),
            'adv_loss': adv_loss.item(),
            'energy_consistency_loss': energy_consistency_loss.item(),
            'pitch_consistency_loss': pitch_consistency_loss.item(),
            'accent_adv_loss': accent_adv_loss.item()
        }

        # Log
        if iteration % 3 == 0:
            print(f"Iteration {iteration}: Loss {loss.item()}")
            print(f"Individual Loss: {individual_loss}")
        
        return loss, individual_loss
