"""HiFi-GAN vocoder fine-tuning script.

Fine-tunes a pretrained HiFi-GAN generator on (predicted_mel, GT_audio) pairs
produced by Daft-Exprt's ``fine_tune.py``, using the full GAN training pipeline
(MPD + MSD discriminators, feature-matching loss, mel-spectrogram L1 loss).

Usage
-----
::

    conda run -n daft_exprt python -m daft_exprt.vocoder.finetune_hifigan \\
        --fine_tuning_dir trainings/hifigan_finetuning_data/fine_tuning_dataset \\
        --gen_checkpoint  src/daft_exprt/vocoder/g_02500000_v1_uni \\
        --disc_checkpoint src/daft_exprt/vocoder/do_02500000 \\
        --output_dir      trainings/hifigan_finetuned \\
        --training_steps  5000
"""

import argparse
import itertools
import logging
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from daft_exprt.vocoder.hifigan import HiFiGANGenerator, DEFAULT_CONFIG
from daft_exprt.vocoder.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from daft_exprt.vocoder.dataset import HiFiGANFinetuneDataset, mel_spectrogram


_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio / model constants (V1 universal, 22 kHz)
# ---------------------------------------------------------------------------
SAMPLING_RATE = 22050
N_FFT = 1024
NUM_MELS = 80
HOP_SIZE = 256
WIN_SIZE = 1024
FMIN = 0
FMAX = 8000
FMAX_FOR_LOSS = None  # full bandwidth for mel L1 loss
SEGMENT_SIZE = 8192


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_generator(checkpoint_path, device):
    """Load pretrained generator *with* weight-norm intact (needed for training)."""
    generator = HiFiGANGenerator(DEFAULT_CONFIG)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('generator', ckpt.get('state_dict', ckpt))
    generator.load_state_dict(state_dict)
    generator.to(device)
    _logger.info('Loaded generator from %s', checkpoint_path)
    return generator


def load_discriminators(disc_checkpoint_path, mpd, msd, device):
    """Optionally load pretrained MPD / MSD weights from a ``do_*`` checkpoint."""
    ckpt = torch.load(disc_checkpoint_path, map_location='cpu')
    mpd.load_state_dict(ckpt['mpd'])
    msd.load_state_dict(ckpt['msd'])
    mpd.to(device)
    msd.to(device)
    _logger.info('Loaded pretrained discriminators from %s (step %s, epoch %s)',
                 disc_checkpoint_path, ckpt.get('steps'), ckpt.get('epoch'))
    return ckpt


def save_checkpoint(generator, step, output_dir, mpd=None, msd=None,
                    optim_g=None, optim_d=None, epoch=0,
                    save_disc=False):
    """Save generator checkpoint.  Optionally save discriminator/optimizer
    state for training resumption (large, ~900 MB)."""
    g_path = os.path.join(output_dir, f'g_{step:08d}')
    torch.save({'generator': generator.state_dict()}, g_path)
    _logger.info('Saved generator checkpoint at step %d -> %s', step, g_path)

    if save_disc and mpd is not None:
        do_path = os.path.join(output_dir, f'do_{step:08d}')
        torch.save({
            'mpd': mpd.state_dict(),
            'msd': msd.state_dict(),
            'optim_g': optim_g.state_dict(),
            'optim_d': optim_d.state_dict(),
            'steps': step,
            'epoch': epoch,
        }, do_path)
        _logger.info('Saved discriminator checkpoint -> %s', do_path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _logger.info('Device: %s', device)

    # ---- models ----------------------------------------------------------
    generator = load_generator(args.gen_checkpoint, device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    disc_ckpt = None
    if args.disc_checkpoint and os.path.isfile(args.disc_checkpoint):
        disc_ckpt = load_discriminators(args.disc_checkpoint, mpd, msd, device)
    else:
        _logger.info('No discriminator checkpoint provided — initialising from scratch')

    # ---- optimisers ------------------------------------------------------
    optim_g = torch.optim.AdamW(
        generator.parameters(), lr=args.learning_rate,
        betas=(args.adam_b1, args.adam_b2))
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        lr=args.learning_rate, betas=(args.adam_b1, args.adam_b2))

    # Optionally restore optimiser state (skip if resuming causes shape issues
    # e.g. when the param count changed because disc was freshly initialised)
    if disc_ckpt is not None and not args.reset_optimizers:
        try:
            optim_g.load_state_dict(disc_ckpt['optim_g'])
            optim_d.load_state_dict(disc_ckpt['optim_d'])
            _logger.info('Restored optimiser states from disc checkpoint')
        except Exception as exc:
            _logger.warning('Could not restore optimiser states (%s) — '
                            'starting optimisers fresh', exc)

    last_epoch = disc_ckpt['epoch'] if disc_ckpt and not args.reset_optimizers else -1
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=args.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=args.lr_decay, last_epoch=last_epoch)

    # ---- data ------------------------------------------------------------
    all_pairs_dataset = HiFiGANFinetuneDataset(
        args.fine_tuning_dir, SEGMENT_SIZE, N_FFT, NUM_MELS, HOP_SIZE,
        WIN_SIZE, SAMPLING_RATE, FMIN, FMAX, split=True, shuffle=True,
        fmax_loss=FMAX_FOR_LOSS, device=device)

    n_total = len(all_pairs_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        all_pairs_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(1234))

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=1,
        pin_memory=True, drop_last=False)

    _logger.info('Dataset: %d train, %d val (total %d pairs)',
                 n_train, n_val, n_total)

    # ---- output dir / tensorboard ----------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    sw = None
    if SummaryWriter is not None:
        log_dir = os.path.join(args.output_dir, 'logs')
        sw = SummaryWriter(log_dir)
        _logger.info('TensorBoard logging to %s', log_dir)
    else:
        _logger.warning('TensorBoard not available — skipping summary logging')

    # ---- training loop ---------------------------------------------------
    generator.train()
    mpd.train()
    msd.train()

    step = 0
    epoch = 0
    start_time = time.time()

    while step < args.training_steps:
        epoch += 1
        for batch in train_loader:
            if step >= args.training_steps:
                break
            start_b = time.time()

            x, y, _fnames, y_mel = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            y = y.unsqueeze(1)  # (B, 1, T)

            # -- generator forward -----------------------------------------
            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1), N_FFT, NUM_MELS, SAMPLING_RATE,
                HOP_SIZE, WIN_SIZE, FMIN, FMAX_FOR_LOSS, center=False)

            # -- discriminator step ----------------------------------------
            optim_d.zero_grad()

            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_f + loss_disc_s
            loss_disc_all.backward()
            optim_d.step()

            # -- generator step --------------------------------------------
            optim_g.zero_grad()

            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)

            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            loss_gen_all.backward()
            optim_g.step()

            step += 1

            # -- stdout logging --------------------------------------------
            if step % args.stdout_interval == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                elapsed = time.time() - start_time
                _logger.info(
                    'Step %d | Gen %.3f | Disc %.3f | Mel L1 %.4f | '
                    '%.2f s/batch | %.1f s elapsed',
                    step, loss_gen_all.item(), loss_disc_all.item(),
                    mel_error, time.time() - start_b, elapsed)

            # -- tensorboard -----------------------------------------------
            if sw is not None and step % args.summary_interval == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                sw.add_scalar('training/gen_loss_total', loss_gen_all.item(), step)
                sw.add_scalar('training/disc_loss_total', loss_disc_all.item(), step)
                sw.add_scalar('training/mel_spec_error', mel_error, step)
                sw.add_scalar('training/loss_mel_x45', loss_mel.item(), step)
                sw.add_scalar('training/loss_gen_f', loss_gen_f.item(), step)
                sw.add_scalar('training/loss_gen_s', loss_gen_s.item(), step)
                sw.add_scalar('training/loss_fm_f', loss_fm_f.item(), step)
                sw.add_scalar('training/loss_fm_s', loss_fm_s.item(), step)

            # -- validation ------------------------------------------------
            if step % args.validation_interval == 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_err = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for j, vbatch in enumerate(val_loader):
                        vx, vy, _vfnames, vy_mel = vbatch
                        vy_g_hat = generator(vx.to(device))
                        vy_mel = vy_mel.to(device)
                        vy_g_hat_mel = mel_spectrogram(
                            vy_g_hat.squeeze(1), N_FFT, NUM_MELS,
                            SAMPLING_RATE, HOP_SIZE, WIN_SIZE, FMIN,
                            FMAX_FOR_LOSS, center=False)
                        val_err += F.l1_loss(vy_mel, vy_g_hat_mel).item()
                        n_val_batches += 1

                        if sw is not None and j < 4:
                            if step == args.validation_interval:
                                sw.add_audio(
                                    f'gt/y_{j}', vy[0], step, SAMPLING_RATE)
                            sw.add_audio(
                                f'generated/y_hat_{j}',
                                vy_g_hat.squeeze(0).cpu(), step, SAMPLING_RATE)

                if n_val_batches > 0:
                    val_err /= n_val_batches
                if sw is not None:
                    sw.add_scalar('validation/mel_spec_error', val_err, step)
                _logger.info('Validation @ step %d | Mel L1 %.4f', step, val_err)
                generator.train()

            # -- checkpointing ---------------------------------------------
            if step % args.checkpoint_interval == 0:
                save_checkpoint(generator, step, args.output_dir,
                                mpd=mpd, msd=msd, optim_g=optim_g,
                                optim_d=optim_d, epoch=epoch,
                                save_disc=args.save_disc)

        scheduler_g.step()
        scheduler_d.step()
        _logger.info('Epoch %d complete (step %d)', epoch, step)

    # Final save
    save_checkpoint(generator, step, args.output_dir,
                    mpd=mpd, msd=msd, optim_g=optim_g,
                    optim_d=optim_d, epoch=epoch,
                    save_disc=args.save_disc)
    if sw is not None:
        sw.close()
    _logger.info('Fine-tuning finished after %d steps.', step)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune HiFi-GAN vocoder on Daft-Exprt TTS outputs')

    # paths
    parser.add_argument('--fine_tuning_dir', type=str, required=True,
                        help='Directory with mel (.npy) / audio (.wav) pairs')
    parser.add_argument('--gen_checkpoint', type=str, required=True,
                        help='Path to pretrained generator checkpoint '
                             '(e.g. g_02500000_v1_uni)')
    parser.add_argument('--disc_checkpoint', type=str, default=None,
                        help='Optional path to pretrained discriminator '
                             'checkpoint (e.g. do_02500000)')
    parser.add_argument('--output_dir', type=str, default='cp_hifigan_ft',
                        help='Directory for checkpoints and TensorBoard logs')

    # training
    parser.add_argument('--training_steps', type=int, default=5000,
                        help='Total fine-tuning steps (default: 5000)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--adam_b1', type=float, default=0.8)
    parser.add_argument('--adam_b2', type=float, default=0.99)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data used for validation (default: 0.1)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--reset_optimizers', action='store_true',
                        help='Ignore optimiser state from disc checkpoint '
                             '(start LR schedule and momentum from scratch)')
    parser.add_argument('--save_disc', action='store_true',
                        help='Save discriminator/optimizer checkpoints '
                             '(~900 MB each) for training resumption. '
                             'Off by default — only the generator is needed '
                             'for inference.')

    # logging
    parser.add_argument('--stdout_interval', type=int, default=5)
    parser.add_argument('--summary_interval', type=int, default=10)
    parser.add_argument('--validation_interval', type=int, default=1000)
    parser.add_argument('--checkpoint_interval', type=int, default=1000)

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
    torch.backends.cudnn.benchmark = True

    train(args)


if __name__ == '__main__':
    main()
