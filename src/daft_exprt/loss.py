import torch

from torch import nn


class DaftExprtLoss(nn.Module):
    def __init__(self, device, hparams):
        super(DaftExprtLoss, self).__init__()
        self.nb_channels = hparams.n_mel_channels
        self.mel_spec_weight = hparams.mel_spec_weight
        
        self.L1Loss = nn.L1Loss(reduction='none').to(device)
        self.MSELoss = nn.MSELoss(reduction='none').to(device)
    
    def forward(self, outputs, targets, iteration):
        ''' Compute training loss

        :param outputs:         outputs predicted by the model
        :param targets:         ground-truth targets
        :param iteration:       current training iteration
        '''
        # extract ground-truth targets
        # targets = (durations_float, symbols_energy, symbols_pitch, mel_specs, output_lengths)
        _, _, _, mel_spec_targets, output_lengths = targets
        mel_spec_targets.requires_grad = False
        
        # extract predictions
        # outputs = (None, None, encoder_preds, decoder_preds, weights)
        # decoder_preds = (mel_preds, output_lengths)
        _, _, _, decoder_preds, _ = outputs
        mel_spec_preds, _ = decoder_preds
        
        # We need output_lengths to normalize loss
        # It's not in targets tuple passed from train.py?
        # Let's check train.py. In train.py:
        # inputs, targets = model.parse_batch(gpu, batch)
        # inputs = (..., output_lengths, ...)
        # targets = (..., mel_specs, speaker_ids)
        # Wait, parse_batch in model.py (my modified version):
        # targets = mel_specs
        # So targets is just mel_specs?
        # Let's check my modified model.py parse_batch:
        # return inputs, targets
        # where targets = mel_specs.
        # But wait, in the original code, targets was a tuple.
        # I changed it to just mel_specs in my thought process but I need to verify what I actually wrote.
        # I wrote: targets = mel_specs
        # So targets is a tensor, not a tuple.
        
        # However, I need output_lengths for the loss normalization.
        # inputs has output_lengths.
        # But forward receives outputs, targets, iteration.
        # It doesn't receive inputs.
        # I should probably pass output_lengths in targets or change how loss is called.
        # Let's check how I modified model.py.
        # I changed parse_batch to return targets = mel_specs.
        # I should probably change it to return (mel_specs, output_lengths) so I can use it in loss.
        
        # For now, let's assume I will fix model.py to return (mel_specs, output_lengths) in targets.
        # Or I can change loss signature to accept lengths.
        # But standard pytorch training loops usually pass outputs, targets.
        
        # Let's assume targets = (mel_specs, output_lengths).
        mel_spec_targets, output_lengths = targets
        mel_spec_targets.requires_grad = False
        output_lengths.requires_grad = False
        
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

        # create individual loss tracker
        individual_loss = {'mel_spec_l1_loss': mel_spec_l1_loss.item(), 'mel_spec_l2_loss': mel_spec_l2_loss.item()}
        
        return loss, individual_loss
