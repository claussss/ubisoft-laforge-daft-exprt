import torch

from torch import nn


class DaftExprtLoss(nn.Module):
    def __init__(self, device, hparams):
        super(DaftExprtLoss, self).__init__()
        self.nb_channels = hparams.n_mel_channels
        self.warmup_steps = getattr(hparams, 'warmup_steps', 10000)
        self.adv_max_weight = getattr(hparams, 'adv_max_weight', 1e-2)
        self.post_mult_weight = getattr(hparams, 'post_mult_weight', 1e-3)
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
        # extract ground-truth targets (prosody is external; only mel, lengths, speaker_ids used for loss)
        # targets = (durations_float, symbols_energy, symbols_pitch, mel_specs, output_lengths, speaker_ids)
        _, _, _, mel_spec_targets, output_lengths, speaker_ids = targets
        mel_spec_targets.requires_grad = False
        speaker_ids.requires_grad = False

        # extract predictions; encoder_preds are pass-through (external prosody), not trained
        # outputs = (speaker_preds, film_params, encoder_preds, decoder_preds, alignments)
        speaker_preds, film_params, _, decoder_preds, _ = outputs
        post_multipliers, _, _, _ = film_params
        mel_spec_preds, output_lengths = decoder_preds
        
        # compute adversarial speaker objective (only if speaker_preds provided)
        if speaker_preds is not None:
            speaker_ce_raw = self.CrossEntropy(speaker_preds, speaker_ids)
            speaker_weight = self.update_adversarial_weight(iteration)
            speaker_loss = speaker_weight * speaker_ce_raw
        else:
            speaker_loss = torch.tensor([0.]).to(mel_spec_preds.device).float()
            speaker_ce_raw = torch.tensor([0.]).to(mel_spec_preds.device).float()
        
        # compute L2 penalized loss on FiLM scalar post-multipliers
        if self.post_mult_weight != 0. and isinstance(post_multipliers, torch.Tensor):
            post_mult_loss = torch.norm(post_multipliers, p=2)
        else:
            post_mult_loss = torch.tensor([0.]).to(mel_spec_preds.device).float()

        # compute mel-spec loss
        mel_spec_l1_loss = self.L1Loss(mel_spec_preds, mel_spec_targets)  # (B, n_mel_channels, T_max)
        mel_spec_l2_loss = self.MSELoss(mel_spec_preds, mel_spec_targets)  # (B, n_mel_channels, T_max)
        # divide by length of each sequence in the batch
        mel_spec_l1_loss = torch.sum(mel_spec_l1_loss, dim=(1, 2)) / (self.nb_channels * output_lengths.float())  # (B, )
        mel_spec_l1_loss = torch.mean(mel_spec_l1_loss)
        mel_spec_l2_loss = torch.sum(mel_spec_l2_loss, dim=(1, 2)) / (self.nb_channels * output_lengths.float())  # (B, )
        mel_spec_l2_loss = torch.mean(mel_spec_l2_loss)

        post_mult_loss = self.post_mult_weight * post_mult_loss
        mel_spec_l1_loss = self.mel_spec_weight * mel_spec_l1_loss
        mel_spec_l2_loss = self.mel_spec_weight * mel_spec_l2_loss

        loss = speaker_loss + post_mult_loss + mel_spec_l1_loss + mel_spec_l2_loss

        individual_loss = {
            'speaker_loss': speaker_loss.item() if isinstance(speaker_loss, torch.Tensor) else speaker_loss,
            'speaker_ce_raw': speaker_ce_raw.item() if isinstance(speaker_ce_raw, torch.Tensor) else speaker_ce_raw,
            'post_mult_loss': post_mult_loss.item() if isinstance(post_mult_loss, torch.Tensor) else post_mult_loss,
            'mel_spec_l1_loss': mel_spec_l1_loss.item(),
            'mel_spec_l2_loss': mel_spec_l2_loss.item()
        }
        
        return loss, individual_loss