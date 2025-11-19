import logging
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # optional dependency
    hf_hub_download = None


_logger = logging.getLogger(__name__)

LRELU_SLOPE = 0.1
DEFAULT_CACHE_DIR = os.path.join(Path.home(), '.cache', 'daft_exprt', 'hifigan')
# publicly mirrored universal HiFi-GAN generator compatible with LJSpeech/22kHz
DEFAULT_CHECKPOINT_URL = 'https://huggingface.co/espnet/kan-bayashi_ljspeech_hifigan/resolve/main/generator.pth?download=1'
DEFAULT_CHECKPOINT_NAME = 'hifigan_ljspeech_generator.pth'

DEFAULT_CONFIG = {
    'sampling_rate': 22050,
    'upsample_rates': [8, 8, 2, 2],
    'upsample_kernel_sizes': [16, 16, 4, 4],
    'upsample_initial_channel': 512,
    'resblock': '1',
    'resblock_kernel_sizes': [3, 7, 11],
    'resblock_dilation_sizes': [
        [1, 3, 5],
        [1, 3, 5],
        [1, 3, 5]
    ],
    'model_in_dim': 80
}


def _download_default_checkpoint(cache_dir=DEFAULT_CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, DEFAULT_CHECKPOINT_NAME)
    if not os.path.isfile(dst):
        if hf_hub_download is not None:
            try:
                repo_id = 'espnet/kan-bayashi_ljspeech_hifigan'
                hf_path = hf_hub_download(repo_id=repo_id, filename='generator.pth')
                shutil.copyfile(hf_path, dst)
                _logger.info('Checkpoint downloaded via huggingface_hub to %s', dst)
                return dst
            except Exception as exc:
                _logger.warning('huggingface_hub download failed (%s). Falling back to direct HTTP.', exc)
        _logger.info('Downloading HiFi-GAN universal vocoder checkpoint from %s', DEFAULT_CHECKPOINT_URL)
        try:
            request = urllib.request.Request(DEFAULT_CHECKPOINT_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request) as resp, open(dst, 'wb') as out_f:
                shutil.copyfileobj(resp, out_f)
        except Exception as exc:
            _logger.error('Failed to download HiFi-GAN checkpoint.\n'
                          'Install huggingface_hub (`pip install huggingface_hub`) or '
                          'provide --vocoder_checkpoint pointing to a local generator.\n'
                          'Original error: %s', exc)
            raise
        _logger.info('Checkpoint downloaded to %s', dst)
    return dst


def init_weights(layer, mean=0.0, std=0.01):
    layer.weight.data.normal_(mean, std)


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=int((kernel_size * d - d) / 2)))
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=int((kernel_size - 1) / 2)))
            for _ in dilations
        ])
        self.num_layers = len(self.convs1)

    def forward(self, x):
        for i in range(self.num_layers):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = self.convs1[i](xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = self.convs2[i](xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=int((kernel_size * d - d) / 2)))
            for d in dilations
        ])
        self.num_layers = len(self.convs)

    def forward(self, x):
        for i in range(self.num_layers):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = self.convs[i](xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)


class HiFiGANGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_kernels = len(config['resblock_kernel_sizes'])
        self.num_upsamples = len(config['upsample_rates'])

        self.conv_pre = weight_norm(nn.Conv1d(config['model_in_dim'], config['upsample_initial_channel'], 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config['upsample_rates'], config['upsample_kernel_sizes'])):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(
                    config['upsample_initial_channel'] // (2 ** i),
                    config['upsample_initial_channel'] // (2 ** (i + 1)),
                    k, u,
                    padding=(k - u) // 2
                )
            ))

        resblock_type = ResBlock1 if config['resblock'] == '1' else ResBlock2
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config['upsample_initial_channel'] // (2 ** (i + 1))
            for k, d in zip(config['resblock_kernel_sizes'], config['resblock_dilation_sizes']):
                self.resblocks.append(resblock_type(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init_weights(m)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs += self.resblocks[idx](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for layer in self.ups:
            remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_post)


class HiFiGanVocoder:
    def __init__(self, checkpoint_path=None, config=None, device='cpu'):
        self.device = torch.device(device)
        if config is None:
            config = DEFAULT_CONFIG
        if checkpoint_path is None:
            checkpoint_path = _download_default_checkpoint()
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.generator = self._load_generator()

    def _load_generator(self):
        generator = HiFiGANGenerator(self.config)
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('generator', checkpoint.get('state_dict', checkpoint))
        generator.load_state_dict(state_dict)
        generator.remove_weight_norm()
        generator.to(self.device)
        generator.eval()
        for param in generator.parameters():
            param.requires_grad = False
        _logger.info('Loaded HiFi-GAN vocoder from %s', self.checkpoint_path)
        return generator

    def infer(self, mel_spec):
        if isinstance(mel_spec, np.ndarray):
            mel = torch.from_numpy(mel_spec).float()
        elif torch.is_tensor(mel_spec):
            mel = mel_spec.float()
        else:
            mel = torch.tensor(mel_spec, dtype=torch.float32)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.dim() == 3 and mel.size(0) != 1:
            raise ValueError('Mel spectrogram batch inference not supported in this helper.')
        mel = mel.to(self.device)
        with torch.no_grad():
            audio = self.generator(mel).squeeze().cpu().numpy()
        audio = np.clip(audio, -1.0, 1.0)
        return audio


def load_hifigan_vocoder(checkpoint_path=None, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return HiFiGanVocoder(checkpoint_path=checkpoint_path, config=DEFAULT_CONFIG, device=device)
