import torch

from torch import nn


class DaftExprtLoss(nn.Module):
    def __init__(self, device, hparams):
        super(DaftExprtLoss, self).__init__()
        self.nb_channels = hparams.n_mel_channels
        self.warmup_steps = getattr(hparams, 'warmup_steps', 10000)
        self.adv_max_weight = getattr(hparams, 'adv_max_weight', 1e-2)
        self.post_mult_weight = getattr(hparams, 'post_mult_weight', 1e-3)
        self.dur_weight = getattr(hparams, 'dur_weight', 1.0)
        self.energy_weight = getattr(hparams, 'energy_weight', 1.0)
        self.pitch_weight = getattr(hparams, 'pitch_weight', 1.0)
        self.mel_spec_weight = getattr(hparams, 'mel_spec_weight', 1.0)
        
        self.L1Loss = nn.L1Loss(reduction='none').to(device)
        self.MSELoss = nn.MSELoss(reduction='none').to(device)
        self.CrossEntropy = nn.CrossEntropyLoss().to(device)
    
    def update_adversarial_weight(self, iteration):
        ''' Update adversarial weight value based on iteration
        '''
        weight_iter = iteration * self.warmup_steps ** -1.5 * self.adv_max_weight / self.warmup_steps ** -0.5
        weight = min(self.adv_max_weight, weight_iter)
        
        return weight
    
    def forward(self, outputs, targets, iteration):
        ''' Compute training loss

        :param outputs:         outputs predicted by the model
        :param targets:         ground-truth targets
        :param iteration:       current training iteration
        '''
        # extract ground-truth targets
        # targets = (durations_float, symbols_energy, symbols_pitch, mel_specs, output_lengths, speaker_ids)
        duration_targets, energy_targets, pitch_targets, mel_spec_targets, output_lengths, speaker_ids = targets
        duration_targets.requires_grad = False
        energy_targets.requires_grad = False
        pitch_targets.requires_grad = False
        mel_spec_targets.requires_grad = False
        speaker_ids.requires_grad = False
        
        # extract predictions
        # outputs = (speaker_preds, film_params, encoder_preds, decoder_preds, alignments)
        speaker_preds, film_params, encoder_preds, decoder_preds, _ = outputs
        post_multipliers, _, _, _ = film_params
        duration_preds, energy_preds, pitch_preds, input_lengths = encoder_preds
        mel_spec_preds, output_lengths = decoder_preds
        
        # compute adversarial speaker objective (only if speaker_preds provided)
        if speaker_preds is not None:
            speaker_loss = self.CrossEntropy(speaker_preds, speaker_ids)
            speaker_weight = self.update_adversarial_weight(iteration)
            speaker_loss = speaker_weight * speaker_loss
        else:
            speaker_loss = torch.tensor([0.]).to(mel_spec_preds.device).float()
        
        # compute L2 penalized loss on FiLM scalar post-multipliers
        if self.post_mult_weight != 0. and isinstance(post_multipliers, torch.Tensor):
            post_mult_loss = torch.norm(post_multipliers, p=2)
        else:
            post_mult_loss = torch.tensor([0.]).to(mel_spec_preds.device).float()
        
        # compute duration loss
        duration_loss = self.MSELoss(duration_preds, duration_targets)  # (B, L_max)
        # divide by length of each sequence in the batch
        duration_loss = torch.sum(duration_loss, dim=1) / input_lengths.float()  # (B, )
        duration_loss = torch.mean(duration_loss)
        
        # compute energy loss
        energy_loss = self.MSELoss(energy_preds, energy_targets)  # (B, L_max)
        # divide by length of each sequence in the batch
        energy_loss = torch.sum(energy_loss, dim=1) / input_lengths.float()  # (B, )
        energy_loss = torch.mean(energy_loss)
        
        # compute pitch loss
        pitch_loss = self.MSELoss(pitch_preds, pitch_targets)  # (B, L_max)
        # divide by length of each sequence in the batch
        pitch_loss = torch.sum(pitch_loss, dim=1) / input_lengths.float()  # (B, )
        pitch_loss = torch.mean(pitch_loss)
        
        # compute mel-spec loss
        mel_spec_l1_loss = self.L1Loss(mel_spec_preds, mel_spec_targets)  # (B, n_mel_channels, T_max)
        mel_spec_l2_loss = self.MSELoss(mel_spec_preds, mel_spec_targets)  # (B, n_mel_channels, T_max)
        # divide by length of each sequence in the batch
        mel_spec_l1_loss = torch.sum(mel_spec_l1_loss, dim=(1, 2)) / (self.nb_channels * output_lengths.float())  # (B, )
        mel_spec_l1_loss = torch.mean(mel_spec_l1_loss)
        mel_spec_l2_loss = torch.sum(mel_spec_l2_loss, dim=(1, 2)) / (self.nb_channels * output_lengths.float())  # (B, )
        mel_spec_l2_loss = torch.mean(mel_spec_l2_loss)

        # add weights
        speaker_loss = speaker_loss
        post_mult_loss = self.post_mult_weight * post_mult_loss
        duration_loss = self.dur_weight * duration_loss
        energy_loss = self.energy_weight * energy_loss
        pitch_loss = self.pitch_weight * pitch_loss
        mel_spec_l1_loss = self.mel_spec_weight * mel_spec_l1_loss
        mel_spec_l2_loss = self.mel_spec_weight * mel_spec_l2_loss
        
        loss = speaker_loss + post_mult_loss + duration_loss + energy_loss + pitch_loss + mel_spec_l1_loss + mel_spec_l2_loss

        # create individual loss tracker
        individual_loss = {
            'speaker_loss': speaker_loss.item() if isinstance(speaker_loss, torch.Tensor) else speaker_loss,
            'post_mult_loss': post_mult_loss.item() if isinstance(post_mult_loss, torch.Tensor) else post_mult_loss,
            'duration_loss': duration_loss.item(),
            'energy_loss': energy_loss.item(),
            'pitch_loss': pitch_loss.item(),
            'mel_spec_l1_loss': mel_spec_l1_loss.item(),
            'mel_spec_l2_loss': mel_spec_l2_loss.item()
        }
        
        return loss, individual_loss
