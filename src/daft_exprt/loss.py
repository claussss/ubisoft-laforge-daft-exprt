import logging
import torch

from torch import nn

from daft_exprt.layers.pitch_predictor import PitchPredictor

_logger = logging.getLogger(__name__)


class DaftExprtLoss(nn.Module):
    """Training loss for Daft-Exprt.

    Components (all gated by hparam weights -- set to 0 to disable):
        1. Mel-spectrogram reconstruction  (L1 + L2)
        2. Adversarial speaker loss        (CrossEntropy with warmup)
        3. FiLM post-multiplier L2 reg
        4. Energy consistency loss          (MSE of smoothed energy extracted from predicted vs target mel-spec)
        5. Pitch consistency loss           (MSE of pitch predicted by frozen PitchPredictor on predicted mel-spec vs GT frames_pitch)
    """

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

        # ── Energy consistency ────────────────────────────────────
        self.energy_consistency_weight = getattr(hparams, 'energy_consistency_weight', 0.0)
        self.avg_pool = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)

        # ── Pitch consistency ─────────────────────────────────────
        self.pitch_consistency_weight = getattr(hparams, 'pitch_consistency_weight', 0.0)
        self.pitch_predictor = None
        pp_path = getattr(hparams, 'pitch_predictor_path', '')
        if self.pitch_consistency_weight > 0 and pp_path:
            _logger.info(f'Loading frozen PitchPredictor from {pp_path}')
            self.pitch_predictor = PitchPredictor(n_mel_channels=hparams.n_mel_channels)
            checkpoint = torch.load(pp_path, map_location=device)
            self.pitch_predictor.load_state_dict(checkpoint)
            self.pitch_predictor.to(device)
            self.pitch_predictor.eval()
            for param in self.pitch_predictor.parameters():
                param.requires_grad = False

    def update_adversarial_weight(self, iteration):
        """Linearly warm up adversarial weight."""
        weight_iter = iteration * self.warmup_steps ** -1.5 * self.adv_max_weight / self.warmup_steps ** -0.5
        return min(self.adv_max_weight, weight_iter)

    def forward(self, outputs, targets, iteration):
        """Compute training loss.

        Args:
            outputs: (speaker_preds, film_params, encoder_preds, decoder_preds, alignments)
            targets: (durations_float, symbols_energy, symbols_pitch,
                      mel_specs, output_lengths, speaker_ids,
                      frames_energy, frames_pitch)          <-- last two added for consistency
            iteration: current training step
        """
        # ── Unpack targets ────────────────────────────────────────
        # Backward compatible: 6 elements = old format, 8 = with frames prosody
        if len(targets) == 8:
            _, _, _, mel_spec_targets, output_lengths, speaker_ids, frames_energy_gt, frames_pitch_gt = targets
        else:
            _, _, _, mel_spec_targets, output_lengths, speaker_ids = targets
            frames_energy_gt = None
            frames_pitch_gt = None
        mel_spec_targets.requires_grad = False
        speaker_ids.requires_grad = False

        # ── Unpack predictions ────────────────────────────────────
        speaker_preds, film_params, _, decoder_preds, _ = outputs
        post_multipliers, _, _, _ = film_params
        mel_spec_preds, output_lengths = decoder_preds

        # ── 1. Adversarial speaker loss ───────────────────────────
        if speaker_preds is not None:
            speaker_ce_raw = self.CrossEntropy(speaker_preds, speaker_ids)
            speaker_weight = self.update_adversarial_weight(iteration)
            speaker_loss = speaker_weight * speaker_ce_raw
        else:
            speaker_loss = torch.tensor([0.]).to(mel_spec_preds.device).float()
            speaker_ce_raw = torch.tensor([0.]).to(mel_spec_preds.device).float()

        # ── 2. FiLM post-multiplier regularisation ────────────────
        if self.post_mult_weight != 0. and isinstance(post_multipliers, torch.Tensor):
            post_mult_loss = torch.norm(post_multipliers, p=2)
        else:
            post_mult_loss = torch.tensor([0.]).to(mel_spec_preds.device).float()

        # ── 3. Mel-spec reconstruction (L1 + L2) ─────────────────
        mel_l1 = self.L1Loss(mel_spec_preds, mel_spec_targets)   # (B, n_mel, T_max)
        mel_l2 = self.MSELoss(mel_spec_preds, mel_spec_targets)  # (B, n_mel, T_max)
        mel_l1 = torch.sum(mel_l1, dim=(1, 2)) / (self.nb_channels * output_lengths.float())
        mel_l1 = torch.mean(mel_l1)
        mel_l2 = torch.sum(mel_l2, dim=(1, 2)) / (self.nb_channels * output_lengths.float())
        mel_l2 = torch.mean(mel_l2)

        post_mult_loss = self.post_mult_weight * post_mult_loss
        mel_l1 = self.mel_spec_weight * mel_l1
        mel_l2 = self.mel_spec_weight * mel_l2

        loss = speaker_loss + post_mult_loss + mel_l1 + mel_l2

        # ── 4. Energy consistency loss ────────────────────────────
        energy_consistency_loss = torch.tensor(0.0, device=loss.device)
        if self.energy_consistency_weight > 0:
            # Energy = L2-norm of linear-scale mel-spec across mel bins
            pred_energy = torch.norm(torch.exp(mel_spec_preds), dim=1)    # (B, T)
            target_energy = torch.norm(torch.exp(mel_spec_targets), dim=1)  # (B, T)

            # Smooth with AvgPool to avoid overly sharp gradients
            pred_energy_s = self.avg_pool(pred_energy.unsqueeze(1)).squeeze(1)
            target_energy_s = self.avg_pool(target_energy.unsqueeze(1)).squeeze(1)

            e_mse = self.MSELoss(pred_energy_s, target_energy_s)  # (B, T)
            mask = (torch.arange(e_mse.size(1), device=e_mse.device)
                    .expand(len(output_lengths), e_mse.size(1)) < output_lengths.unsqueeze(1))
            e_mse = e_mse * mask.float()

            energy_consistency_loss = torch.sum(e_mse) / torch.sum(output_lengths).float()
            loss = loss + self.energy_consistency_weight * energy_consistency_loss

        # ── 5. Pitch consistency loss ─────────────────────────────
        pitch_consistency_loss = torch.tensor(0.0, device=loss.device)
        if self.pitch_predictor is not None and frames_pitch_gt is not None:
            pred_pitch = self.pitch_predictor(mel_spec_preds)  # (B, T)

            p_mse = self.MSELoss(pred_pitch, frames_pitch_gt)  # (B, T)

            # Mask: valid length AND voiced frames only (GT pitch == 0 means unvoiced)
            len_mask = (torch.arange(p_mse.size(1), device=p_mse.device)
                        .expand(len(output_lengths), p_mse.size(1)) < output_lengths.unsqueeze(1))
            voiced_mask = (frames_pitch_gt != 0.0)
            combined_mask = len_mask & voiced_mask

            p_mse = p_mse * combined_mask.float()
            pitch_consistency_loss = torch.sum(p_mse) / (torch.sum(combined_mask.float()) + 1e-5)
            loss = loss + self.pitch_consistency_weight * pitch_consistency_loss

        # ── Individual loss tracking ──────────────────────────────
        individual_loss = {
            'speaker_loss': speaker_loss.item() if isinstance(speaker_loss, torch.Tensor) else speaker_loss,
            'speaker_ce_raw': speaker_ce_raw.item() if isinstance(speaker_ce_raw, torch.Tensor) else speaker_ce_raw,
            'post_mult_loss': post_mult_loss.item() if isinstance(post_mult_loss, torch.Tensor) else post_mult_loss,
            'mel_spec_l1_loss': mel_l1.item(),
            'mel_spec_l2_loss': mel_l2.item(),
            'energy_consistency_loss': energy_consistency_loss.item(),
            'pitch_consistency_loss': pitch_consistency_loss.item(),
        }

        return loss, individual_loss
