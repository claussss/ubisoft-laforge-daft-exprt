import numpy as np
import torch

from collections import namedtuple

from torch import nn
from torch.autograd import Function
from torch.distributions import Normal
from torch.nn.parameter import Parameter

from daft_exprt.extract_features import duration_to_integer


def get_mask_from_lengths(lengths):
    ''' Create a masked tensor from given lengths

    :param lengths:     torch.tensor of size (B, ) -- lengths of each example

    :return mask: torch.tensor of size (B, max_length) -- the masked tensor
    '''
    max_len = torch.max(lengths)
    ids = torch.arange(0, max_len).to(lengths.device, non_blocking=True).long()
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


class LinearNorm(nn.Module):
    ''' Linear Norm Module:
        - Linear Layer
    '''
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        ''' Forward function of Linear Norm
            x = (*, in_dim)
        '''
        x = self.linear_layer(x)  # (*, out_dim)
        
        return x


class ConvNorm1D(nn.Module):
    ''' Conv Norm 1D Module:
        - Conv 1D
    '''
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))
    
    def forward(self, x):
        ''' Forward function of Conv Norm 1D
            x = (B, L, in_channels)
        '''
        x = x.transpose(1, 2)  # (B, in_channels, L)
        x = self.conv(x)  # (B, out_channels, L)
        x = x.transpose(1, 2)  # (B, L, out_channels)
        
        return x


class ConvNorm2D(nn.Module):
    ''' Conv Norm 2D Module:
        - Conv 2D
    '''
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        ''' Forward function of Conv Norm 2D:
            x = (B, H, W, in_channels)
        '''
        x = x.permute(0, 3, 1, 2)  # (B, in_channels, H, W)
        x = self.conv(x)  # (B, out_channels, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, out_channels)
        
        return x


class PositionalEncoding(nn.Module):
    ''' Positional Encoding Module:
        - Sinusoidal Positional Embedding
    '''
    def __init__(self, embed_dim, max_len=5000, timestep=10000.):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim 
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-np.log(timestep) / self.embed_dim))  # (embed_dim // 2, )
        self.pos_enc = torch.FloatTensor(max_len, self.embed_dim).zero_()  # (max_len, embed_dim)
        self.pos_enc[:, 0::2] = torch.sin(pos * div_term)
        self.pos_enc[:, 1::2] = torch.cos(pos * div_term)
    
    def forward(self, x):
        ''' Forward function of Positional Encoding:
            x = (B, N) -- Long or Int tensor
        '''
        # initialize tensor
        nb_frames_max = int(torch.max(torch.cumsum(x, dim=1)))
        pos_emb = torch.FloatTensor(x.size(0), nb_frames_max, self.embed_dim).zero_()  # (B, nb_frames_max, embed_dim)
        pos_emb = pos_emb.to(x.device, non_blocking=True).float()  # (B, nb_frames_max, embed_dim)
        
        # can be used for absolute or relative positioning
        for line_idx in range(x.size(0)):
            pos_idx = []
            for column_idx in range(x.size(1)):
                idx = x[line_idx, column_idx]
                pos_idx.extend([i for i in range(idx)])
            emb = self.pos_enc[pos_idx]  # (nb_frames, embed_dim)
            pos_emb[line_idx, :emb.size(0), :] = emb
        
        return pos_emb


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention Module:
        - Multi-Head Attention
            A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser and I. Polosukhin
            "Attention is all you need",
            in NeurIPS, 2017.
        - Dropout
        - Residual Connection 
        - Layer Normalization
    '''
    def __init__(self, hparams):
        super(MultiHeadAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(hparams.hidden_embed_dim,
                                                          hparams.attn_nb_heads,
                                                          hparams.attn_dropout)
        self.dropout = nn.Dropout(hparams.attn_dropout)
        self.layer_norm = nn.LayerNorm(hparams.hidden_embed_dim)
    
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        ''' Forward function of Multi-Head Attention:
            query = (B, L_max, hidden_embed_dim)
            key = (B, T_max, hidden_embed_dim)
            value = (B, T_max, hidden_embed_dim)
            key_padding_mask = (B, T_max) if not None
            attn_mask = (L_max, T_max) if not None
        '''
        # compute multi-head attention
        # attn_outputs = (L_max, B, hidden_embed_dim)
        # attn_weights = (B, L_max, T_max)
        attn_outputs, attn_weights = self.multi_head_attention(query.transpose(0, 1),
                                                               key.transpose(0, 1),
                                                               value.transpose(0, 1),
                                                               key_padding_mask=key_padding_mask,
                                                               attn_mask=attn_mask)
        attn_outputs = attn_outputs.transpose(0, 1)  # (B, L_max, hidden_embed_dim)
        # apply dropout
        attn_outputs = self.dropout(attn_outputs)  # (B, L_max, hidden_embed_dim)
        # add residual connection and perform layer normalization
        attn_outputs = self.layer_norm(attn_outputs + query)  # (B, L_max, hidden_embed_dim)

        return attn_outputs, attn_weights


class PositionWiseConvFF(nn.Module):
    ''' Position Wise Convolutional Feed-Forward Module:
        - 2x Conv 1D with ReLU
        - Dropout
        - Residual Connection 
        - Layer Normalization
        - FiLM conditioning (if film_params is not None)
    '''
    def __init__(self, hparams):
        super(PositionWiseConvFF, self).__init__()
        self.convs = nn.Sequential(
            ConvNorm1D(hparams.hidden_embed_dim, hparams.conv_channels,
                       kernel_size=hparams.conv_kernel, stride=1,
                       padding=int((hparams.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.ReLU(),
            ConvNorm1D(hparams.conv_channels, hparams.hidden_embed_dim,
                       kernel_size=hparams.conv_kernel, stride=1,
                       padding=int((hparams.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='linear'),
            nn.Dropout(hparams.conv_dropout)
        )
        self.layer_norm = nn.LayerNorm(hparams.hidden_embed_dim)
    
    def forward(self, x, film_params):
        ''' Forward function of PositionWiseConvFF:
            x = (B, L_max, hidden_embed_dim)
            film_params = (B, nb_film_params) or None
        '''
        # pass through convs
        ff = self.convs(x)  # (B, L_max, hidden_embed_dim)
        # add residual connection and perform layer normalization
        outputs = self.layer_norm(ff + x)  # (B, L_max, hidden_embed_dim)
        # add FiLM transformation
        if film_params is not None:
            nb_gammas = int(film_params.size(1) / 2)
            assert(nb_gammas == outputs.size(2))
            gammas = film_params[:, :nb_gammas].unsqueeze(1)  # (B, 1, hidden_embed_dim)
            betas = film_params[:, nb_gammas:].unsqueeze(1)  # (B, 1, hidden_embed_dim)
            outputs = gammas * outputs + betas  # (B, L_max, hidden_embed_dim)
        
        return outputs


class FFTBlock(nn.Module):
    ''' FFT Block Module:
        - Multi-Head Attention
        - Position Wise Convolutional Feed-Forward
        - FiLM conditioning (if film_params is not None)
    '''
    def __init__(self, hparams):
        super(FFTBlock, self).__init__()
        self.attention = MultiHeadAttention(hparams)
        self.feed_forward = PositionWiseConvFF(hparams)
    
    def forward(self, x, film_params, mask):
        ''' Forward function of FFT Block:
            x = (B, L_max, hidden_embed_dim)
            film_params = (B, nb_film_params) or None
            mask = (B, L_max)
        '''
        # attend
        attn_outputs, _ = self.attention(x, x, x, key_padding_mask=mask)  # (B, L_max, hidden_embed_dim)
        attn_outputs = attn_outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, hidden_embed_dim)
        # feed-forward pass
        outputs = self.feed_forward(attn_outputs, film_params)
        outputs = outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, hidden_embed_dim)
        
        return outputs


class SpeakerFiLMGenerator(nn.Module):
    ''' Generates FiLM parameters from speaker embedding for FrameDecoder only.
    
    Replaces the ProsodyEncoder + StyleAdapter with a simple speaker-only system.
    '''
    def __init__(self, hparams):
        super(SpeakerFiLMGenerator, self).__init__()
        self.n_speakers = hparams.n_speakers
        hidden_dim = hparams.phoneme_encoder['hidden_embed_dim']
        self.post_mult_weight = getattr(hparams, 'post_mult_weight', 1e-3)
        
        # Speaker embedding
        self.spk_embedding = nn.Embedding(self.n_speakers, hidden_dim)
        torch.nn.init.xavier_uniform_(self.spk_embedding.weight.data)
        
        # FiLM for frame_decoder only (no phoneme encoder, no prosody predictor FiLM)
        nb_decoder_blocks = hparams.frame_decoder['nb_blocks']
        decoder_hidden_dim = hparams.phoneme_encoder['hidden_embed_dim']
        
        self.module_params = {
            'frame_decoder': (nb_decoder_blocks, decoder_hidden_dim)
        }
        
        nb_tot_film_params = sum(nb_blocks * channels for nb_blocks, channels in self.module_params.values())
        
        # Single linear layer for FiLM prediction (like original ProsodyEncoder)
        self.gammas_predictor = LinearNorm(hidden_dim, nb_tot_film_params, w_init_gain='linear')
        self.betas_predictor = LinearNorm(hidden_dim, nb_tot_film_params, w_init_gain='linear')
        
        # Initialize L2 penalized scalar post-multipliers
        if self.post_mult_weight > 0:
            nb_post_multipliers = sum(params[0] for params in self.module_params.values())
            self.post_multipliers = Parameter(torch.empty(2, nb_post_multipliers))
            nn.init.xavier_uniform_(self.post_multipliers, gain=nn.init.calculate_gain('linear'))
        else:
            self.post_multipliers = 1.0
    
    def forward(self, speaker_ids):
        ''' Forward pass: speaker_ids (B,) -> FiLM params for decoder
        '''
        spk_emb = self.spk_embedding(speaker_ids)  # (B, hidden_dim)
        
        gammas = self.gammas_predictor(spk_emb)  # (B, nb_tot_film_params)
        betas = self.betas_predictor(spk_emb)    # (B, nb_tot_film_params)
        
        film_params = {}
        column_idx, block_idx = 0, 0
        
        for module_name, (nb_blocks, channels) in self.module_params.items():
            module_nb_params = nb_blocks * channels
            module_gammas = gammas[:, column_idx:column_idx + module_nb_params].view(-1, nb_blocks, channels)
            module_betas = betas[:, column_idx:column_idx + module_nb_params].view(-1, nb_blocks, channels)
            
            if self.post_mult_weight > 0:
                gamma_post = self.post_multipliers[0, block_idx:block_idx + nb_blocks].unsqueeze(0).unsqueeze(-1)
                beta_post = self.post_multipliers[1, block_idx:block_idx + nb_blocks].unsqueeze(0).unsqueeze(-1)
                module_gammas = gamma_post * module_gammas + 1  # delta regime
                module_betas = beta_post * module_betas
            else:
                module_gammas = module_gammas + 1
            
            film_params[module_name] = torch.cat((module_gammas, module_betas), dim=2)
            block_idx += nb_blocks
            column_idx += module_nb_params
        
        return film_params


class LocalProsodyPredictor(nn.Module):
    ''' Local Prosody Predictor Module:
        - 2x Conv 1D (per block)
        - NO FiLM conditioning (independent module)
        - Linear projection to predict [duration, energy, pitch]
    '''
    def __init__(self, hparams):
        super(LocalProsodyPredictor, self).__init__()
        embed_dim = hparams.phoneme_encoder['hidden_embed_dim']
        Tuple = namedtuple('Tuple', hparams.local_prosody_predictor)
        hparams_lpp = Tuple(**hparams.local_prosody_predictor)
        
        # conv1D blocks
        blocks = []
        for idx in range(hparams_lpp.nb_blocks):
            in_channels = embed_dim if idx == 0 else hparams_lpp.conv_channels
            convs = nn.Sequential(
                ConvNorm1D(in_channels, hparams_lpp.conv_channels,
                           kernel_size=hparams_lpp.conv_kernel, stride=1,
                           padding=int((hparams_lpp.conv_kernel - 1) / 2),
                           dilation=1, w_init_gain='relu'),
                nn.ReLU(),
                nn.LayerNorm(hparams_lpp.conv_channels),
                nn.Dropout(hparams_lpp.conv_dropout),
                ConvNorm1D(hparams_lpp.conv_channels, hparams_lpp.conv_channels,
                           kernel_size=hparams_lpp.conv_kernel, stride=1,
                           padding=int((hparams_lpp.conv_kernel - 1) / 2),
                           dilation=1, w_init_gain='relu'),
                nn.ReLU(),
                nn.LayerNorm(hparams_lpp.conv_channels),
                nn.Dropout(hparams_lpp.conv_dropout)
            )
            blocks.append(convs)
        self.blocks = nn.ModuleList(blocks)
        
        # linear projection for prosody prediction (duration, energy, pitch)
        self.projection = LinearNorm(hparams_lpp.conv_channels, 3, w_init_gain='linear')
    
    def forward(self, x, input_lengths):
        ''' Forward function of Local Prosody Predictor:
            x = (B, L_max, hidden_embed_dim)
            input_lengths = (B, )
            
        Returns: duration_preds, energy_preds, pitch_preds (all B, L_max)
        '''
        # pass through blocks
        for block in self.blocks:
            x = block(x)  # (B, L_max, conv_channels)
        
        # mask tensor
        mask = ~get_mask_from_lengths(input_lengths)  # (B, L_max)
        x = x.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, conv_channels)
        
        # predict prosody params and mask tensor
        prosody_preds = self.projection(x)  # (B, L_max, 3)
        prosody_preds = prosody_preds.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, 3)
        
        # extract prosody params
        durations = prosody_preds[:, :, 0]  # (B, L_max)
        energies = prosody_preds[:, :, 1]   # (B, L_max)
        pitch = prosody_preds[:, :, 2]      # (B, L_max)
        
        return durations, energies, pitch


class GaussianUpsamplingModule(nn.Module):
    ''' Gaussian Upsampling Module:
        - Duration Projection
        - Energy Projection
        - Pitch Projection
        - Ranges Projection Layer
        - Gaussian Upsampling
    '''
    def __init__(self, hparams):
        super(GaussianUpsamplingModule, self).__init__()
        embed_dim = hparams.phoneme_encoder['hidden_embed_dim']
        Tuple = namedtuple('Tuple', hparams.gaussian_upsampling_module)
        hparams_gum = Tuple(**hparams.gaussian_upsampling_module)
        
        # duration, energy and pitch projection layers
        self.duration_projection = ConvNorm1D(1, embed_dim, kernel_size=hparams_gum.conv_kernel,
                                              stride=1, padding=int((hparams_gum.conv_kernel - 1) / 2),
                                              dilation=1, w_init_gain='linear')
        self.energy_projection = ConvNorm1D(1, embed_dim, kernel_size=hparams_gum.conv_kernel,
                                            stride=1, padding=int((hparams_gum.conv_kernel - 1) / 2),
                                            dilation=1, w_init_gain='linear')
        self.pitch_projection = ConvNorm1D(1, embed_dim, kernel_size=hparams_gum.conv_kernel,
                                           stride=1, padding=int((hparams_gum.conv_kernel - 1) / 2),
                                           dilation=1, w_init_gain='linear')
        # ranges predictor
        self.projection = nn.Sequential(
            LinearNorm(embed_dim, 1, w_init_gain='relu'),
            nn.Softplus()
        )
    
    def forward(self, x, durations_float, durations_int, energies, pitch, input_lengths):
        ''' Forward function of Gaussian Upsampling Module:
            x = (B, L_max, hidden_embed_dim)
            durations_float = (B, L_max)
            durations_int = (B, L_max)
            energies = (B, L_max)
            pitch = (B, L_max)
            input_lengths = (B, )
        '''
        # project durations
        durations = durations_float.unsqueeze(2)  # (B, L_max, 1)
        durations = self.duration_projection(durations)  # (B, L_max, hidden_embed_dim)
        
        # project energies
        energies_proj = energies.unsqueeze(2)  # (B, L_max, 1)
        energies_proj = self.energy_projection(energies_proj)  # (B, L_max, hidden_embed_dim)
        
        # project pitch
        pitch_proj = pitch.unsqueeze(2)  # (B, L_max, 1)
        pitch_proj = self.pitch_projection(pitch_proj)  # (B, L_max, hidden_embed_dim)
        
        # add energy and pitch to encoded input symbols
        x = x + energies_proj + pitch_proj  # (B, L_max, hidden_embed_dim)
        
        # predict ranges for each symbol and mask tensor
        # use mask_value = 1. because ranges will be used as stds in Gaussian upsampling
        range_inputs = x + durations  # (B, L_max, hidden_embed_dim) 
        ranges = self.projection(range_inputs)  # (B, L_max, 1)
        ranges = ranges.squeeze(2)  # (B, L_max)
        mask = ~get_mask_from_lengths(input_lengths)  # (B, L_max)
        ranges = ranges.masked_fill(mask, 1)  # (B, L_max)
        
        # STABILITY: Clamp ranges to prevent NaN/invalid stds
        ranges = torch.clamp(ranges, min=1e-3)
        
        # perform Gaussian upsampling
        # compute Gaussian means
        means = durations_int.float() / 2  # (B, L_max)
        cumsum = torch.cumsum(durations_int, dim=1)  # (B, L_max)
        means[:, 1:] += cumsum[:, :-1]  # (B, L_max)
        
        # compute Gaussian distributions
        means = means.unsqueeze(-1)  # (B, L_max, 1)
        stds = ranges.unsqueeze(-1)  # (B, L_max, 1)
        
        # STABILITY: Additional check for NaN/Inf
        means = torch.nan_to_num(means, nan=0.0, posinf=1e6, neginf=-1e6)
        stds = torch.nan_to_num(stds, nan=1.0, posinf=1e6, neginf=1e-3)
        stds = torch.clamp(stds, min=1e-3)
        
        gaussians = Normal(means, stds)  # (B, L_max, 1)
        
        # create frames idx tensor
        nb_frames_max = torch.max(cumsum)  # T_max
        frames_idx = torch.FloatTensor([i + 0.5 for i in range(nb_frames_max)])  # (T_max, )
        frames_idx = frames_idx.to(x.device, non_blocking=True).float()  # (T_max, )
        
        # compute probs
        probs = torch.exp(gaussians.log_prob(frames_idx))  # (B, L_max, T_max)
        # apply mask to set probs out of sequence length to 0
        probs = probs.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, T_max)
        
        # compute weights
        weights = probs / (torch.sum(probs, dim=1, keepdim=True) + 1e-20)  # (B, L_max, T_max)
        
        # compute upsampled embedding
        x_upsamp = torch.sum(x.unsqueeze(-1) * weights.unsqueeze(2), dim=1)  # (B, hidden_embed_dim, T_max)
        x_upsamp = x_upsamp.permute(0, 2, 1)  # (B, T_max, hidden_embed_dim)
        
        return x_upsamp, weights


class FrameDecoder(nn.Module):
    ''' Frame Decoder Module:
        - Positional Encoding
        - 4x FFT Blocks with FiLM conditioning (speaker only)
        - Linear projection
    '''
    def __init__(self, hparams, is_training=True):
        super(FrameDecoder, self).__init__()
        nb_mels = hparams.n_mel_channels
        embed_dim = hparams.phoneme_encoder['hidden_embed_dim']
        
        self.hidden_dim = embed_dim
        self.nb_mels = nb_mels
        
        hparams.frame_decoder['hidden_embed_dim'] = embed_dim
        Tuple = namedtuple('Tuple', hparams.frame_decoder)
        hparams_fd = Tuple(**hparams.frame_decoder)
        
        # positional encoding
        self.pos_enc = PositionalEncoding(embed_dim)
        
        # FFT blocks
        blocks = []
        for _ in range(hparams_fd.nb_blocks):
            blocks.append(FFTBlock(hparams_fd))
        self.blocks = nn.ModuleList(blocks)
        
        # linear projection for mel-spec prediction
        self.projection = LinearNorm(embed_dim, nb_mels, w_init_gain='linear')
        
        # Accent adaptation modules - ONLY created for inference
        self._is_training = is_training
        if not is_training:
            self._init_adaptation_modules(hparams)
    
    def _init_adaptation_modules(self, hparams):
        """Initialize adaptation modules for inference only."""
        nb_mels = self.nb_mels
        embed_dim = self.hidden_dim
        
        # Tier toggles (default: all disabled)
        self.enable_tier_a = False
        self.enable_tier_b = False
        self.enable_tier_c = False
        
        # Tier A: Global Accent Filter
        self.global_accent_filter = GlobalAccentFilter(
            n_mels=nb_mels, s_scale_max=0.2, s_bias_max=0.05
        )
        
        # Tier B: Shared Accent FiLM
        self.accent_gamma_raw = nn.Parameter(torch.zeros(embed_dim))
        self.accent_beta_raw = nn.Parameter(torch.zeros(embed_dim))
        self.accent_s_max = 0.3
        self.accent_s_raw = nn.Parameter(torch.tensor(-1.4))
        
        # Tier C: Shared Pointwise Adapter
        self.pointwise_adapter = PointwiseAdapter(
            hidden_dim=embed_dim, bottleneck=16, g_max=0.3
        )
    
    def forward(self, x, film_params, output_lengths):
        ''' Forward function of Decoder Embedding:
            x = (B, T_max, hidden_embed_dim)
            film_params = (B, nb_blocks, nb_film_params)
            output_lengths = (B, )
        '''
        # compute positional encoding
        pos = self.pos_enc(output_lengths.unsqueeze(1))  # (B, T_max, hidden_embed_dim)
        # create mask
        mask = ~get_mask_from_lengths(output_lengths)  # (B, T_max)
        # add and mask
        x = x + pos  # (B, T_max, hidden_embed_dim)
        x = x.masked_fill(mask.unsqueeze(2), 0)  # (B, T_max, hidden_embed_dim)
        
        # pass through FFT blocks
        for idx, block in enumerate(self.blocks):
            x = block(x, film_params[:, idx, :], mask)
        
        # predict mel-spec frames and mask tensor
        mel_specs = self.projection(x)  # (B, T_max, nb_mels)
        mel_specs = mel_specs.masked_fill(mask.unsqueeze(2), 0)  # (B, T_max, nb_mels)
        mel_specs = mel_specs.transpose(1, 2)  # (B, nb_mels, T_max)
        
        # Apply adaptation if available and enabled (inference only)
        if hasattr(self, 'enable_tier_a') and self.enable_tier_a:
            mel_specs = self.global_accent_filter(mel_specs)
        
        return mel_specs


class PhonemeEncoder(nn.Module):
    ''' Phoneme Encoder Module:
        - Symbols Embedding
        - Positional Encoding
        - 4x FFT Blocks (NO FiLM - independent encoder)
    '''
    def __init__(self, hparams):
        super(PhonemeEncoder, self).__init__()
        n_symbols = hparams.n_symbols
        embed_dim = hparams.phoneme_encoder['hidden_embed_dim']
        Tuple = namedtuple('Tuple', hparams.phoneme_encoder)
        hparams_pe = Tuple(**hparams.phoneme_encoder)
        
        # symbols embedding and positional encoding
        self.symbols_embedding = nn.Embedding(n_symbols, embed_dim)
        torch.nn.init.xavier_uniform_(self.symbols_embedding.weight.data)
        self.pos_enc = PositionalEncoding(embed_dim)
        
        # FFT blocks
        blocks = []
        for _ in range(hparams_pe.nb_blocks):
            blocks.append(FFTBlock(hparams_pe))
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x, input_lengths):
        ''' Forward function of Phoneme Encoder:
            x = (B, L_max)
            input_lengths = (B, )
        
        No FiLM conditioning - independent encoder
        '''
        # compute symbols embedding
        x = self.symbols_embedding(x)  # (B, L_max, hidden_embed_dim)
        # compute positional encoding
        pos = self.pos_enc(input_lengths.unsqueeze(1))  # (B, L_max, hidden_embed_dim)
        # create mask
        mask = ~get_mask_from_lengths(input_lengths)  # (B, L_max)
        # add and mask
        x = x + pos  # (B, L_max, hidden_embed_dim)
        x = x.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, hidden_embed_dim)
        
        # pass through FFT blocks (NO FiLM)
        for block in self.blocks:
            x = block(x, None, mask)  # None = no FiLM
        
        return x


# ============ INFERENCE-ONLY MODULES ============
# These are kept for inference/adaptation but not used during training

class GlobalAccentFilter(nn.Module):
    """Tier A: Global Accent Filter applied after mel projection.
    
    Learns a per-frequency affine transform for accent coloration.
    INFERENCE ONLY - not created during training.
    """
    def __init__(self, n_mels=80, s_scale_max=0.2, s_bias_max=0.05):
        super().__init__()
        self.n_mels = n_mels
        self.s_scale_max = s_scale_max
        self.s_bias_max = s_bias_max
        
        self.a_raw = nn.Parameter(torch.zeros(n_mels))
        self.b_raw = nn.Parameter(torch.zeros(n_mels))
        self.s_scale_raw = nn.Parameter(torch.tensor(-1.1))
        self.s_bias_raw = nn.Parameter(torch.tensor(-1.4))
        
    def forward(self, mel):
        s_scale = self.s_scale_max * torch.sigmoid(self.s_scale_raw)
        s_bias = self.s_bias_max * torch.sigmoid(self.s_bias_raw)
        a = 1 + s_scale * torch.tanh(self.a_raw)
        b = s_bias * torch.tanh(self.b_raw)
        return a.view(1, -1, 1) * mel + b.view(1, -1, 1)


class PointwiseAdapter(nn.Module):
    """Tier C: Pointwise Bottleneck Adapter for decoder FFN.
    
    INFERENCE ONLY - not created during training.
    """
    def __init__(self, hidden_dim=256, bottleneck=16, g_max=0.3):
        super().__init__()
        self.g_max = g_max
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_dim)
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)
        nn.init.xavier_uniform_(self.up.weight, gain=0.1)
        
    def forward(self, ff):
        ad = self.up(torch.relu(self.down(ff)))
        g = self.g_max * torch.sigmoid(self.gate)
        return ff + g * ad


class DaftExprt(nn.Module):
    ''' DaftExprt model - Simplified architecture
    
    Architecture:
        PhonemeEncoder (no FiLM) -> LocalProsodyPredictor (no FiLM) 
        -> GaussianUpsampling -> FrameDecoder (FiLM from speaker only)
    '''
    def __init__(self, hparams, is_training=True):
        super(DaftExprt, self).__init__()
        self.n_speakers = hparams.n_speakers
        self.hidden_embed_dim = hparams.phoneme_encoder['hidden_embed_dim']
        
        # Speaker FiLM Generator (replaces ProsodyEncoder + StyleAdapter)
        self.speaker_film = SpeakerFiLMGenerator(hparams)
        
        # Core Components
        self.phoneme_encoder = PhonemeEncoder(hparams)
        self.prosody_predictor = LocalProsodyPredictor(hparams)
        self.gaussian_upsampling = GaussianUpsamplingModule(hparams)
        self.frame_decoder = FrameDecoder(hparams, is_training=is_training)
    
    def parse_batch(self, device, batch):
        ''' Parse input batch
        '''
        # Batch format: (symbols, durations_float, durations_int, symbols_energy, symbols_pitch, 
        #                input_lengths, frames_energy, frames_pitch, mel_specs, output_lengths, 
        #                speaker_ids, feature_dirs, feature_files)
        symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths, \
            frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids, feature_dirs, feature_files = batch
        
        # transfer tensors to specified device
        symbols = symbols.to(device, non_blocking=True).long()
        durations_float = durations_float.to(device, non_blocking=True).float()
        durations_int = durations_int.to(device, non_blocking=True).long()
        symbols_energy = symbols_energy.to(device, non_blocking=True).float()
        symbols_pitch = symbols_pitch.to(device, non_blocking=True).float()
        input_lengths = input_lengths.to(device, non_blocking=True).long()
        frames_energy = frames_energy.to(device, non_blocking=True).float()
        frames_pitch = frames_pitch.to(device, non_blocking=True).float()
        mel_specs = mel_specs.to(device, non_blocking=True).float()
        output_lengths = output_lengths.to(device, non_blocking=True).long()
        speaker_ids = speaker_ids.to(device, non_blocking=True).long()
        
        # create inputs and targets
        inputs = (symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths,
                  frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids)
        targets = (durations_float, symbols_energy, symbols_pitch, mel_specs, output_lengths, speaker_ids)
        
        return inputs, targets
    
    def forward(self, inputs):
        ''' Forward function of DaftExprt
        
        Flow:
        1. Generate FiLM params from speaker ID only
        2. Encode phonemes (NO FiLM - independent)
        3. Predict prosody (NO FiLM - independent)
        4. Gaussian upsampling (uses GT prosody for training)
        5. Frame decoder (WITH speaker FiLM)
        '''
        # Extract inputs
        symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths, \
            frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids = inputs
        input_lengths, output_lengths = input_lengths.detach(), output_lengths.detach()
        
        # 1. Generate FiLM params from speaker ID only
        film_params_dict = self.speaker_film(speaker_ids)
        # Returns: {'frame_decoder': (B, nb_blocks, nb_film_params)}
        
        # 2. Encode Phonemes (NO FiLM - independent)
        enc_outputs = self.phoneme_encoder(symbols, input_lengths)  # (B, L_max, hidden_embed_dim)
        
        # 3. Predict prosody (NO FiLM - independent)
        duration_preds, energy_preds, pitch_preds = self.prosody_predictor(enc_outputs, input_lengths)
        
        # 4. Gaussian Upsampling (uses GT prosody for training)
        symbols_upsamp, weights = self.gaussian_upsampling(
            enc_outputs, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths
        )  # (B, T_max, hidden_embed_dim)
        
        # 5. Frame Decoder (WITH speaker FiLM)
        mel_spec_preds = self.frame_decoder(
            symbols_upsamp, 
            film_params_dict['frame_decoder'], 
            output_lengths
        )  # (B, n_mel_channels, T_max)
        
        # Parse outputs (format matching original loss expectations)
        film_params = [self.speaker_film.post_multipliers, None, None, film_params_dict['frame_decoder']]
        encoder_preds = [duration_preds, energy_preds, pitch_preds, input_lengths]
        decoder_preds = [mel_spec_preds, output_lengths]
        alignments = weights
        
        # No speaker predictions (no adversarial loss)
        speaker_preds = None
        
        return speaker_preds, film_params, encoder_preds, decoder_preds, alignments
    
    def get_int_durations(self, duration_preds, hparams):
        ''' Convert float durations to integer frame durations
        '''
        # min float duration to have at least one mel-spec frame attributed to the symbol
        fft_length = hparams.filter_length / hparams.sampling_rate
        dur_min = fft_length / 2
        # set duration under min duration to 0.
        duration_preds[duration_preds < dur_min] = 0.
        # convert to int durations for each element in the batch
        durations_int = torch.LongTensor(duration_preds.size(0), duration_preds.size(1)).zero_()
        for line_idx in range(duration_preds.size(0)):
            end_prev, symbols_idx, durations_float = 0., [], []
            for symbol_id in range(duration_preds.size(1)):
                symb_dur = duration_preds[line_idx, symbol_id].item()
                if symb_dur != 0.:
                    symbols_idx.append(symbol_id)
                    durations_float.append([end_prev, end_prev + symb_dur])
                    end_prev += symb_dur
            int_durs = torch.LongTensor(duration_to_integer(durations_float, hparams))
            durations_int[line_idx, symbols_idx] = int_durs
        durations_int = durations_int.to(duration_preds.device, non_blocking=True).long()
        
        return duration_preds, durations_int
    
    def pitch_shift(self, pitch_preds, pitch_factors, hparams, speaker_ids):
        ''' Pitch shift pitch predictions
        '''
        zero_idxs = (pitch_preds == 0.).nonzero()
        for line_idx in range(pitch_preds.size(0)):
            speaker_id = speaker_ids[line_idx].item()
            pitch_mean = hparams.stats[f'spk {speaker_id}']['pitch']['mean']
            pitch_std = hparams.stats[f'spk {speaker_id}']['pitch']['std']
            pitch_preds[line_idx] = torch.exp(pitch_std * pitch_preds[line_idx] + pitch_mean)
            pitch_preds[line_idx] += pitch_factors[line_idx]
            pitch_preds[line_idx] = (torch.log(pitch_preds[line_idx]) - pitch_mean) / pitch_std
        pitch_preds[zero_idxs[:, 0], zero_idxs[:, 1]] = 0.
        
        return pitch_preds
    
    def pitch_multiply(self, pitch_preds, pitch_factors):
        ''' Apply multiply transform to pitch prediction with respect to the mean
        '''
        for line_idx in range(pitch_preds.size(0)):
            non_zero_idxs = pitch_preds[line_idx].nonzero()
            zero_idxs = (pitch_preds[line_idx] == 0.).nonzero()
            mean_pitch = torch.mean(pitch_preds[line_idx, non_zero_idxs])
            pitch_deviation = pitch_preds[line_idx] - mean_pitch
            pitch_deviation *= pitch_factors[line_idx]
            pitch_preds[line_idx] += pitch_deviation
            pitch_preds[line_idx, zero_idxs] = 0.
        
        return pitch_preds
    
    def inference(self, inputs, pitch_transform, hparams):
        ''' Inference function of DaftExprt
        '''
        # Unpack inputs
        symbols, dur_factors, energy_factors, pitch_factors, input_lengths, \
            energy_refs, pitch_refs, mel_spec_refs, ref_lengths, speaker_ids = inputs
        
        # 1. Generate FiLM params from speaker ID only
        film_params_dict = self.speaker_film(speaker_ids)
        
        # 2. Encode Phonemes (NO FiLM)
        enc_outputs = self.phoneme_encoder(symbols, input_lengths)
        
        # 3. Predict prosody (NO FiLM)
        duration_preds, energy_preds, pitch_preds = self.prosody_predictor(enc_outputs, input_lengths)
        
        # Apply factors
        duration_preds = duration_preds * dur_factors
        duration_preds, durations_int = self.get_int_durations(duration_preds, hparams)
        
        energy_preds = energy_preds * energy_factors
        energy_preds[durations_int == 0] = 0.
        pitch_preds[durations_int == 0] = 0.
        
        if pitch_transform == 'add':
            pitch_preds = self.pitch_shift(pitch_preds, pitch_factors, hparams, speaker_ids)
        elif pitch_transform == 'multiply':
            pitch_preds = self.pitch_multiply(pitch_preds, pitch_factors)
        else:
            raise NotImplementedError
        
        # 4. Gaussian Upsampling
        symbols_upsamp, weights = self.gaussian_upsampling(
            enc_outputs, duration_preds, durations_int, energy_preds, pitch_preds, input_lengths
        )
        
        # Get output lengths
        output_lengths = torch.sum(durations_int, dim=1)
        output_lengths = output_lengths.to(symbols_upsamp.device, non_blocking=True).long()
        output_lengths[output_lengths == 0] = 1
        
        assert(torch.max(output_lengths) == symbols_upsamp.size(1))
        
        # 5. Frame Decoder (WITH speaker FiLM)
        mel_spec_preds = self.frame_decoder(
            symbols_upsamp, 
            film_params_dict['frame_decoder'], 
            output_lengths
        )
        
        # Parse outputs
        encoder_preds = [duration_preds, durations_int, energy_preds, pitch_preds, input_lengths]
        decoder_preds = [mel_spec_preds, output_lengths]
        alignments = weights
        
        return encoder_preds, decoder_preds, alignments
