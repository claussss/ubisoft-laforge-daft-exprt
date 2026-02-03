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


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    ''' Gradient Reversal Layer
            Y. Ganin, V. Lempitsky,
            "Unsupervised Domain Adaptation by Backpropagation",
            in ICML, 2015.
        Forward pass is the identity function
        In the backward pass, upstream gradients are multiplied by -lambda (i.e. gradient are reversed)
    '''
    def __init__(self, hparams):
        super(GradientReversal, self).__init__()
        self.lambda_ = hparams.lambda_reversal

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


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
            film_params = (B, nb_film_params)
        '''
        outputs = self.convs(x)  # (B, L_max, hidden_embed_dim)
        outputs = self.layer_norm(outputs + x)  # (B, L_max, hidden_embed_dim)

        if film_params is not None:
            nb_gammas = int(film_params.size(1) / 2)
            assert nb_gammas == outputs.size(2)
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
            film_params = (B, nb_film_params)
            mask = (B, L_max)
        '''
        attn_outputs, _ = self.attention(x, x, x, key_padding_mask=mask)  # (B, L_max, hidden_embed_dim)
        attn_outputs = attn_outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, hidden_embed_dim)
        outputs = self.feed_forward(attn_outputs, film_params)
        outputs = outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, hidden_embed_dim)
        return outputs


class AccentEncoder(nn.Module):
    """ Accent Encoder: Extracts global accent style from mel-spectrograms + prosody """
    def __init__(self, hparams):
        super(AccentEncoder, self).__init__()
        nb_mels = hparams.n_mel_channels
        Tuple = namedtuple('Tuple', hparams.accent_encoder)
        hparams_ae = Tuple(**hparams.accent_encoder)

        self.pos_enc = PositionalEncoding(hparams_ae.hidden_embed_dim)
        self.energy_embedding = ConvNorm1D(1, hparams_ae.hidden_embed_dim, kernel_size=hparams_ae.conv_kernel,
                                           stride=1, padding=int((hparams_ae.conv_kernel - 1) / 2), dilation=1, w_init_gain='linear')
        self.pitch_embedding = ConvNorm1D(1, hparams_ae.hidden_embed_dim, kernel_size=hparams_ae.conv_kernel,
                                          stride=1, padding=int((hparams_ae.conv_kernel - 1) / 2), dilation=1, w_init_gain='linear')

        self.convs = nn.Sequential(
            ConvNorm1D(nb_mels, hparams_ae.conv_channels, kernel_size=hparams_ae.conv_kernel, stride=1,
                       padding=int((hparams_ae.conv_kernel - 1) / 2), dilation=1, w_init_gain='relu'),
            nn.ReLU(), nn.LayerNorm(hparams_ae.conv_channels), nn.Dropout(hparams_ae.conv_dropout),
            ConvNorm1D(hparams_ae.conv_channels, hparams_ae.conv_channels, kernel_size=hparams_ae.conv_kernel, stride=1,
                       padding=int((hparams_ae.conv_kernel - 1) / 2), dilation=1, w_init_gain='relu'),
            nn.ReLU(), nn.LayerNorm(hparams_ae.conv_channels), nn.Dropout(hparams_ae.conv_dropout),
            ConvNorm1D(hparams_ae.conv_channels, hparams_ae.hidden_embed_dim, kernel_size=hparams_ae.conv_kernel, stride=1,
                       padding=int((hparams_ae.conv_kernel - 1) / 2), dilation=1, w_init_gain='relu'),
            nn.ReLU(), nn.LayerNorm(hparams_ae.hidden_embed_dim), nn.Dropout(hparams_ae.conv_dropout)
        )

        blocks = [FFTBlock(hparams_ae) for _ in range(hparams_ae.nb_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, frames_energy, frames_pitch, mel_specs, output_lengths):
        pos = self.pos_enc(output_lengths.unsqueeze(1))
        energy = self.energy_embedding(frames_energy.unsqueeze(2))
        pitch = self.pitch_embedding(frames_pitch.unsqueeze(2))
        outputs = self.convs(mel_specs.transpose(1, 2))
        mask = ~get_mask_from_lengths(output_lengths)
        outputs = (outputs + energy + pitch + pos).masked_fill(mask.unsqueeze(2), 0)
        for block in self.blocks:
            outputs = block(outputs, None, mask)
        accent_embedding = torch.sum(outputs, dim=1) / output_lengths.unsqueeze(1)
        return accent_embedding


class SpeakerClassifier(nn.Module):
    """ Speaker Classifier with GRL for accent disentanglement """
    def __init__(self, hparams):
        super(SpeakerClassifier, self).__init__()
        nb_speakers = hparams.n_speakers
        embed_dim = hparams.accent_encoder['hidden_embed_dim']
        self.classifier = nn.Sequential(
            GradientReversal(hparams),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'), nn.ReLU(),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'), nn.ReLU(),
            LinearNorm(embed_dim, nb_speakers, w_init_gain='linear')
        )

    def forward(self, x):
        return self.classifier(x)


class StyleAdapter(nn.Module):
    """ Style Adapter: Generates FiLM parameters from summed accent+speaker embeddings

    Matches original ProsodyEncoder architecture:
    - Single linear layer for FiLM prediction (not 2-layer MLP)
    - FiLM conditioning on phoneme_encoder and frame_decoder only (no upsampling)
    - Input is summed embeddings (accent_emb + spk_emb), not concatenated
    """
    def __init__(self, hparams):
        super(StyleAdapter, self).__init__()
        # Input is now summed embedding (same dim as each component)
        input_dim = hparams.accent_encoder['hidden_embed_dim']
        self.post_mult_weight = hparams.post_mult_weight

        decoder_hidden_dim = hparams.phoneme_encoder['hidden_embed_dim']

        # Only condition phoneme_encoder and frame_decoder (no upsampling, like original)
        self.module_params = {
            'phoneme_encoder': (hparams.phoneme_encoder['nb_blocks'], hparams.phoneme_encoder['hidden_embed_dim']),
            'frame_decoder': (hparams.frame_decoder['nb_blocks'], decoder_hidden_dim)
        }

        nb_tot_film_params = sum(nb_blocks * channels for nb_blocks, channels in self.module_params.values())

        # Single linear layer (like original ProsodyEncoder)
        self.gammas_predictor = LinearNorm(input_dim, nb_tot_film_params, w_init_gain='linear')
        self.betas_predictor = LinearNorm(input_dim, nb_tot_film_params, w_init_gain='linear')

        # Initialize L2 penalized scalar post-multipliers
        if self.post_mult_weight > 0:
            nb_post_multipliers = sum(params[0] for params in self.module_params.values())
            self.post_multipliers = Parameter(torch.empty(2, nb_post_multipliers))
            nn.init.xavier_uniform_(self.post_multipliers, gain=nn.init.calculate_gain('linear'))
        else:
            self.post_multipliers = 1.0

    def forward(self, combined_emb):
        """ Forward pass: combined_emb is summed accent+speaker embedding (B, hidden_dim) """
        gammas = self.gammas_predictor(combined_emb)  # (B, nb_tot_film_params)
        betas = self.betas_predictor(combined_emb)    # (B, nb_tot_film_params)

        film_params = {}
        column_idx, block_idx = 0, 0

        for module_name, (nb_blocks, channels) in self.module_params.items():
            module_nb_params = nb_blocks * channels
            module_gammas = gammas[:, column_idx:column_idx + module_nb_params].view(-1, nb_blocks, channels)
            module_betas = betas[:, column_idx:column_idx + module_nb_params].view(-1, nb_blocks, channels)

            if self.post_mult_weight > 0:
                gamma_post = self.post_multipliers[0, block_idx:block_idx + nb_blocks].unsqueeze(0).unsqueeze(-1)
                beta_post = self.post_multipliers[1, block_idx:block_idx + nb_blocks].unsqueeze(0).unsqueeze(-1)
                module_gammas = gamma_post * module_gammas + 1
                module_betas = beta_post * module_betas
            else:
                module_gammas = module_gammas + 1

            film_params[module_name] = torch.cat((module_gammas, module_betas), dim=2)
            block_idx += nb_blocks
            column_idx += module_nb_params

        return film_params


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
        gum_dict = dict(hparams.gaussian_upsampling_module)
        Tuple = namedtuple('Tuple', gum_dict.keys())
        hparams_gum = Tuple(**gum_dict)
        self.use_concatenation = getattr(hparams_gum, 'use_concatenation', False)

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

    def forward(self, x, durations_float, durations_int, energies, pitch, input_lengths, film_params=None):
        ''' Forward function of Gaussian Upsampling Module:
            x = (B, L_max, hidden_embed_dim)
            durations_float = (B, L_max)
            durations_int = (B, L_max)
            energies = (B, L_max)
            pitch = (B, L_max)
            input_lengths = (B, )
            film_params = (B, 3, 2*hidden_embed_dim) or None - FiLM params for 3 projections
        '''
        embed_dim = x.size(2)

        # project durations
        durations = durations_float.unsqueeze(2)  # (B, L_max, 1)
        durations = self.duration_projection(durations)  # (B, L_max, hidden_embed_dim)

        # Apply FiLM to duration projection
        if film_params is not None:
            fp_dur = film_params[:, 0, :]  # (B, 2*embed_dim)
            gamma = fp_dur[:, :embed_dim].unsqueeze(1)  # (B, 1, embed_dim)
            beta = fp_dur[:, embed_dim:].unsqueeze(1)  # (B, 1, embed_dim)
            durations = gamma * durations + beta
            durations = torch.nn.functional.relu(durations)

        # project energies
        energies = energies.unsqueeze(2)  # (B, L_max, 1)
        energies = self.energy_projection(energies)  # (B, L_max, hidden_embed_dim)

        # Apply FiLM to energy projection
        if film_params is not None:
            fp_nrg = film_params[:, 1, :]
            gamma = fp_nrg[:, :embed_dim].unsqueeze(1)
            beta = fp_nrg[:, embed_dim:].unsqueeze(1)
            energies = gamma * energies + beta
            energies = torch.nn.functional.relu(energies)

        # project pitch
        pitch = pitch.unsqueeze(2)  # (B, L_max, 1)
        pitch = self.pitch_projection(pitch)  # (B, L_max, hidden_embed_dim)

        # Apply FiLM to pitch projection
        if film_params is not None:
            fp_f0 = film_params[:, 2, :]
            gamma = fp_f0[:, :embed_dim].unsqueeze(1)
            beta = fp_f0[:, embed_dim:].unsqueeze(1)
            pitch = gamma * pitch + beta
            pitch = torch.nn.functional.relu(pitch)

        # add energy and pitch to encoded input symbols
        if self.use_concatenation:
            x_combined = torch.cat([x, energies, pitch], dim=2)
            x_summed = x + energies + pitch
        else:
            x = x + energies + pitch  # (B, L_max, hidden_embed_dim)
            x_combined = x
            x_summed = x

        # predict ranges for each symbol and mask tensor
        range_inputs = x_summed + durations  # (B, L_max, hidden_embed_dim)
        ranges = self.projection(range_inputs)  # (B, L_max, 1)
        ranges = ranges.squeeze(2)  # (B, L_max)
        mask = ~get_mask_from_lengths(input_lengths) # (B, L_max)
        ranges = ranges.masked_fill(mask, 1)  # (B, L_max)

        # STABILITY: Clamp ranges to prevent NaN/invalid stds in Normal distribution
        ranges = torch.clamp(ranges, min=1e-3)  # Minimum std of 0.001

        # perform Gaussian upsampling
        means = durations_int.float() / 2  # (B, L_max)
        cumsum = torch.cumsum(durations_int, dim=1)  # (B, L_max)
        means[:, 1:] += cumsum[:, :-1]  # (B, L_max)
        means = means.unsqueeze(-1)  # (B, L_max, 1)
        stds = ranges.unsqueeze(-1)  # (B, L_max, 1)

        means = torch.nan_to_num(means, nan=0.0, posinf=1e6,  neginf=-1e6)
        stds = torch.nan_to_num(stds, nan=1.0, posinf=1e6, neginf=1e-3)
        stds = torch.clamp(stds, min=1e-3)  # Ensure positive

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
        x_upsamp = torch.sum(x.unsqueeze(-1) * weights.unsqueeze(2), dim=1)  # (B, input_dim, T_max)
        x_upsamp = x_upsamp.permute(0, 2, 1)  # (B, T_max, input_dim)

        return x_upsamp, weights


class FrameDecoder(nn.Module):
    ''' Frame Decoder Module:
        - Positional Encoding
        - 4x FFT Blocks with FiLM conditioning
        - Linear projection
    '''
    def __init__(self, hparams, is_training=True):
        super(FrameDecoder, self).__init__()
        nb_mels = hparams.n_mel_channels
        embed_dim = hparams.phoneme_encoder['hidden_embed_dim']

        # Check if input_dim is provided in hparams (for concatenation support)
        if hasattr(hparams, 'frame_decoder_input_dim'):
            embed_dim = hparams.frame_decoder_input_dim
        else:
            embed_dim = hparams.phoneme_encoder['hidden_embed_dim']

        self.hidden_dim = embed_dim
        self.nb_mels = nb_mels

        # "Highway" Architecture:
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

    def forward(self, x, film_params, output_lengths):
        ''' Forward function of Decoder Embedding:
            x = (B, T_max, hidden_embed_dim)
            film_params = (B, nb_blocks, nb_film_params)
            output_lengths = (B, )
        '''
        pos = self.pos_enc(output_lengths.unsqueeze(1))  # (B, T_max, hidden_embed_dim)
        mask = ~get_mask_from_lengths(output_lengths)  # (B, T_max)
        x = x + pos
        x = x.masked_fill(mask.unsqueeze(2), 0)  # (B, T_max, hidden_embed_dim)

        for idx, block in enumerate(self.blocks):
            x = block(x, film_params[:, idx, :], mask)
        mel_specs = self.projection(x)  # (B, T_max, nb_mels)
        mel_specs = mel_specs.masked_fill(mask.unsqueeze(2), 0)  # (B, T_max, nb_mels)
        mel_specs = mel_specs.transpose(1, 2)  # (B, nb_mels, T_max)
        return mel_specs


class PhonemeEncoder(nn.Module):
    ''' Phoneme Encoder Module:
        - Symbols Embedding
        - Positional Encoding
        - 4x FFT Blocks with FiLM conditioning
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

    def forward(self, x, film_params, input_lengths):
        ''' Forward function of Phoneme Encoder:
            x = (B, L_max)
            film_params = (B, nb_blocks, nb_film_params) or None
            input_lengths = (B, )
        '''
        # compute symbols embedding
        x = self.symbols_embedding(x)  # (B, L_max, hidden_embed_dim)
        # compute positional encoding
        pos = self.pos_enc(input_lengths.unsqueeze(1))  # (B, L_max, hidden_embed_dim)
        # create mask
        mask = ~get_mask_from_lengths(input_lengths) # (B, L_max)
        # add and mask
        x = x + pos  # (B, L_max, hidden_embed_dim)
        x = x.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, hidden_embed_dim)
        # pass through FFT blocks
        for idx, block in enumerate(self.blocks):
            fp = film_params[:, idx, :] if film_params is not None else None
            x = block(x, fp, mask)  # (B, L_max, hidden_embed_dim)

        return x



class AccentEncoder(nn.Module):
    ''' Accent Encoder Module (extracted from ProsodyEncoder):
        - Energy/Pitch/Mel-Spec Embeddings
        - 4x FFT Blocks
        - Pooled Output
    '''
    def __init__(self, hparams):
        super(AccentEncoder, self).__init__()
        nb_mels = hparams.n_mel_channels
        
        # We need to construct a specific hparams tuple for the accent encoder if it exists, 
        # otherwise fallback to phoneme_encoder params or similar defaults.
        # Checkpoint analysis suggests 'accent_encoder' key exists in hparams.
        if hasattr(hparams, 'accent_encoder'):
            ae_params = hparams.accent_encoder
        else:
            # Fallback to phoneme_encoder config if not found (or define default)
            ae_params = hparams.phoneme_encoder
            
        Tuple = namedtuple('Tuple', ae_params)
        hparams_ae = Tuple(**ae_params)

        # positional encoding
        self.pos_enc = PositionalEncoding(hparams_ae.hidden_embed_dim)
        
        # energy embedding
        self.energy_embedding = ConvNorm1D(1, hparams_ae.hidden_embed_dim, kernel_size=hparams_ae.conv_kernel,
                                           stride=1, padding=int((hparams_ae.conv_kernel - 1) / 2),
                                           dilation=1, w_init_gain='linear')
        # pitch embedding
        self.pitch_embedding = ConvNorm1D(1, hparams_ae.hidden_embed_dim, kernel_size=hparams_ae.conv_kernel,
                                          stride=1, padding=int((hparams_ae.conv_kernel - 1) / 2),
                                          dilation=1, w_init_gain='linear')
        
        # mel-spec pre-net convolutions (Assuming same structure as ProsodyEncoder)
        self.convs = nn.Sequential(
            ConvNorm1D(nb_mels, hparams_ae.conv_channels,
                       kernel_size=hparams_ae.conv_kernel, stride=1,
                       padding=int((hparams_ae.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.ReLU(),
            nn.LayerNorm(hparams_ae.conv_channels),
            nn.Dropout(hparams_ae.conv_dropout),
            ConvNorm1D(hparams_ae.conv_channels, hparams_ae.conv_channels,
                       kernel_size=hparams_ae.conv_kernel, stride=1,
                       padding=int((hparams_ae.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.ReLU(),
            nn.LayerNorm(hparams_ae.conv_channels),
            nn.Dropout(hparams_ae.conv_dropout),
            ConvNorm1D(hparams_ae.conv_channels, hparams_ae.hidden_embed_dim,
                       kernel_size=hparams_ae.conv_kernel, stride=1,
                       padding=int((hparams_ae.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.ReLU(),
            nn.LayerNorm(hparams_ae.hidden_embed_dim),
            nn.Dropout(hparams_ae.conv_dropout)
        )
        
        # FFT blocks
        blocks = []
        for _ in range(hparams_ae.nb_blocks):
            blocks.append(FFTBlock(hparams_ae))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, frames_energy, frames_pitch, mel_specs, output_lengths):
        ''' 
        frames_energy: (B, T_max)
        frames_pitch: (B, T_max)
        mel_specs: (B, n_mels, T_max)
        output_lengths: (B,)
        '''
        # compute positional encoding
        pos = self.pos_enc(output_lengths.unsqueeze(1))  # (B, T_max, hidden_embed_dim)
        
        # encode energy sequence
        frames_energy = frames_energy.unsqueeze(2)  # (B, T_max, 1)
        energy = self.energy_embedding(frames_energy)  # (B, T_max, hidden_embed_dim)
        
        # encode pitch sequence
        frames_pitch = frames_pitch.unsqueeze(2)  # (B, T_max, 1)
        pitch = self.pitch_embedding(frames_pitch)  # (B, T_max, hidden_embed_dim)
        
        # pass through convs
        mel_specs = mel_specs.transpose(1, 2)  # (B, T_max, nb_mels)
        outputs = self.convs(mel_specs)  # (B, T_max, hidden_embed_dim)
        
        # create mask
        mask = ~get_mask_from_lengths(output_lengths) # (B, T_max)
        
        # add encodings and mask tensor
        outputs = outputs + energy + pitch + pos  # (B, T_max, hidden_embed_dim)
        outputs = outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, T_max, hidden_embed_dim)
        
        # pass through FFT blocks
        for _, block in enumerate(self.blocks):
            outputs = block(outputs, None, mask)  # (B, T_max, hidden_embed_dim)
            
        # average pooling on the whole time sequence
        # (B, hidden_embed_dim)
        outputs = torch.sum(outputs, dim=1) / output_lengths.unsqueeze(1)
        
        return outputs


class StyleAdapter(nn.Module):
    ''' Style Adapter Module (extracted from ProsodyEncoder):
        - Predicts FiLM parameters (Gamma, Beta) from style embedding
    '''
    def __init__(self, hparams):
        super(StyleAdapter, self).__init__()
        
        # Determine hidden_embed_dim (usually same as phoneme/accent encoder)
        if hasattr(hparams, 'accent_encoder'):
             hidden_dim = hparams.accent_encoder['hidden_embed_dim']
        else:
             hidden_dim = hparams.phoneme_encoder['hidden_embed_dim']
             
        self.module_params = {
            'phoneme_encoder': (hparams.phoneme_encoder['nb_blocks'], hparams.phoneme_encoder['hidden_embed_dim']),
            'frame_decoder': (hparams.frame_decoder['nb_blocks'], hparams.phoneme_encoder['hidden_embed_dim'])
        }
        
        # Fix for FrameDecoder hidden dim if inconsistent
        if 'frame_decoder' in self.module_params:
             # In original ProsodyEncoder: 'decoder': (hparams.frame_decoder['nb_blocks'], hparams.phoneme_encoder['hidden_embed_dim'])
             self.module_params['frame_decoder'] = (hparams.frame_decoder['nb_blocks'], hparams.phoneme_encoder['hidden_embed_dim'])
        
        nb_tot_film_params = 0
        for _, module_params in self.module_params.items():
            nb_blocks, conv_channels = module_params
            nb_tot_film_params += nb_blocks * conv_channels

        self.gammas_predictor = LinearNorm(hidden_dim, nb_tot_film_params, w_init_gain='linear')
        self.betas_predictor = LinearNorm(hidden_dim, nb_tot_film_params, w_init_gain='linear')
        
        # initialize L2 penalized scalar post-multipliers
        # one (gamma, beta) scalar post-multiplier per FiLM layer, i.e per block
        if hasattr(hparams, 'post_mult_weight') and hparams.post_mult_weight != 0.:
            self.post_mult_weight = hparams.post_mult_weight
            nb_post_multipliers = 0
            for _, module_params in self.module_params.items():
                nb_blocks, _ = module_params
                nb_post_multipliers += nb_blocks
            self.post_multipliers = Parameter(torch.empty(2, nb_post_multipliers))  # (2, nb_post_multipliers)
            nn.init.xavier_uniform_(self.post_multipliers, gain=nn.init.calculate_gain('linear'))
        else:
            self.post_mult_weight = 0.
            self.post_multipliers = 1.

    def forward(self, style_embedding):
        ''' 
        style_embedding: (B, hidden_dim)
        Returns: dictionary of film_params per module
        '''
        gammas = self.gammas_predictor(style_embedding)  # (B, nb_tot_film_params)
        betas = self.betas_predictor(style_embedding)    # (B, nb_tot_film_params)
        
        film_params_dict = {}
        column_idx, block_idx = 0, 0
        
        for module_name, module_params in self.module_params.items():
            nb_blocks, conv_channels = module_params
            module_nb_film_params = nb_blocks * conv_channels
            
            module_gammas = gammas[:, column_idx: column_idx + module_nb_film_params]
            module_betas = betas[:, column_idx: column_idx + module_nb_film_params]
            
            B = module_gammas.size(0)
            module_gammas = module_gammas.view(B, nb_blocks, -1)
            module_betas = module_betas.view(B, nb_blocks, -1)
            
            # Post-multipliers logic
            if self.post_mult_weight != 0.:
                gamma_post = self.post_multipliers[0, block_idx: block_idx + nb_blocks]
                gamma_post = gamma_post.unsqueeze(0).unsqueeze(-1)
                
                beta_post = self.post_multipliers[1, block_idx: block_idx + nb_blocks]
                beta_post = beta_post.unsqueeze(0).unsqueeze(-1)
            else:
                gamma_post = self.post_multipliers
                beta_post = self.post_multipliers
                
            module_gammas = gamma_post * module_gammas + 1
            module_betas = beta_post * module_betas
            
            module_film_params = torch.cat((module_gammas, module_betas), dim=2)
            film_params_dict[module_name] = module_film_params
            
            block_idx += nb_blocks
            column_idx += module_nb_film_params
            
        return film_params_dict


class SpeakerClassifier(nn.Module):
    ''' Speaker Classifier Module:
        - 3x Linear Layers with ReLU
        - Used for Adversarial Training
    '''
    def __init__(self, hparams):
        super(SpeakerClassifier, self).__init__()
        # Match checkpoint: trained with n_speakers classes (e.g. 12), not n_speakers - 1
        nb_speakers = hparams.n_speakers
        embed_dim = hparams.phoneme_encoder['hidden_embed_dim'] # Assuming same dim
        
        self.classifier = nn.Sequential(
            GradientReversal(hparams),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, nb_speakers, w_init_gain='linear')
        )
    
    def forward(self, x):
        return self.classifier(x)

class DaftExprt(nn.Module):

    ''' DaftExprt model from J. Zaïdi, H. Seuté, B. van Niekerk, M.A. Carbonneau
        "DaftExprt: Robust Prosody Transfer Across Speakers for Expressive Speech Synthesis"
        arXiv:2108.02271, 2021.
    '''
    def __init__(self, hparams, is_training=True):
        super(DaftExprt, self).__init__()
        self.n_speakers = hparams.n_speakers
        self.hidden_embed_dim = hparams.phoneme_encoder['hidden_embed_dim']

        # Accent Transfer Components (NEW)
        self.accent_encoder = AccentEncoder(hparams)
        self.speaker_classifier = SpeakerClassifier(hparams)
        self.style_adapter = StyleAdapter(hparams)

        # Core Components
        self.phoneme_encoder = PhonemeEncoder(hparams)
        self.gaussian_upsampling = GaussianUpsamplingModule(hparams)

        self.frame_decoder = FrameDecoder(hparams, is_training=is_training)

        # Speaker: ECAPA external embeddings only (zero-shot). No lookup table.
        external_emb_dim = getattr(hparams, 'external_emb_dim', 192)
        self.spk_projection = LinearNorm(external_emb_dim, self.hidden_embed_dim)

    def parse_batch(self, device, batch):
        ''' Parse input batch. Requires 14 elements including spk_embs (ECAPA precomputed). '''
        if len(batch) != 14:
            raise ValueError(
                f"Batch must have 14 elements (including speaker embeddings). Got {len(batch)}. "
                "Run training.py pre_process to compute ECAPA embeddings and ensure .spk_emb.npy files exist."
            )
        symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths, \
            frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids, feature_dirs, feature_files, spk_embs = batch
        spk_embs = spk_embs.to(device, non_blocking=True).float()

        # transfer tensors to specified device
        symbols = symbols.to(device, non_blocking=True).long()                        # (B, L_max)
        durations_float = durations_float.to(device, non_blocking=True).float()       # (B, L_max)
        durations_int = durations_int.to(device, non_blocking=True).long()            # (B, L_max)
        symbols_energy = symbols_energy.to(device, non_blocking=True).float()         # (B, L_max)
        symbols_pitch = symbols_pitch.to(device, non_blocking=True).float()           # (B, L_max)
        input_lengths = input_lengths.to(device, non_blocking=True).long()            # (B, )
        frames_energy = frames_energy.to(device, non_blocking=True).float()           # (B, T_max)
        frames_pitch = frames_pitch.to(device, non_blocking=True).float()             # (B, T_max)
        mel_specs = mel_specs.to(device, non_blocking=True).float()                   # (B, n_mel_channels, T_max)
        output_lengths = output_lengths.to(device, non_blocking=True).long()          # (B, )
        speaker_ids = speaker_ids.to(device, non_blocking=True).long()                # (B, )

        # create inputs and targets
        inputs = (symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths,
                  frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids, spk_embs)
        targets = (durations_float, symbols_energy, symbols_pitch, mel_specs, output_lengths, speaker_ids)

        return inputs, targets

    def forward(self, inputs, external_accent_emb=None, external_spk_emb=None):
        """
        Forward function of DaftExprt with Accent Transfer
        """
        if len(inputs) != 12:
            raise ValueError(f"inputs must have 12 elements (including spk_embs). Got {len(inputs)}.")
        symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths, \
            frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids, spk_embs = inputs

        # 1. Speaker embedding (ECAPA only)
        if external_spk_emb is not None:
            spk_emb = external_spk_emb
        else:
            if spk_embs is None:
                raise ValueError("Speaker embeddings (spk_embs) required. Precompute ECAPA and provide .spk_emb.npy in data.")
            spk_emb = torch.nn.functional.normalize(spk_embs, p=2, dim=-1)
            spk_emb = self.spk_projection(spk_emb)

        # 2. Encode accent (self-reference: use current batch features unless external provided)
        if external_accent_emb is not None:
            accent_emb = external_accent_emb
        else:
            accent_emb = self.accent_encoder(
                frames_energy, frames_pitch, mel_specs, output_lengths
            )  # (B, hidden_embed_dim)

        # 3. Speaker classification (for adversarial training with GRL)
        speaker_preds = self.speaker_classifier(accent_emb)  # (B, nb_speakers)

        # 4. Fuse accent + speaker and generate FiLM parameters (SUMMATION like original)
        combined_emb = accent_emb + spk_emb  # (B, hidden_dim)
        film_params_dict = self.style_adapter(combined_emb)

        # 5. Encode Phonemes (with accent conditioning)
        enc_outputs = self.phoneme_encoder(
            symbols,
            film_params_dict['phoneme_encoder'],
            input_lengths
        )  # (B, L_max, hidden_embed_dim)

        # 6. Gaussian Upsampling (no FiLM)
        x, weights = self.gaussian_upsampling(
            enc_outputs, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths,
            film_params=None
        )  # (B, T_max, hidden_embed_dim)

        # 7. Frame Decoder (with accent conditioning)
        mel_preds = self.frame_decoder(
            x,
            film_params_dict['frame_decoder'],
            output_lengths
        )  # (B, n_mel_channels, T_max)

        # Adapt to loss expectations (speaker_preds used for adversarial speaker loss)
        film_params = [self.style_adapter.post_multipliers, None, None, film_params_dict['frame_decoder']]
        encoder_preds = [durations_float, symbols_energy, symbols_pitch, input_lengths]
        decoder_preds = [mel_preds, output_lengths]
        alignments = weights

        return speaker_preds, film_params, encoder_preds, decoder_preds, alignments

    def get_int_durations(self, duration_preds, hparams):
        ''' Convert float durations to integer frame durations
        '''
        # min float duration to have at least one mel-spec frame attributed to the symbol
        fft_length = hparams.filter_length / hparams.sampling_rate
        dur_min = fft_length / 2
        # set duration under min duration to 0.
        duration_preds[duration_preds < dur_min] = 0.  # (B, L_max)
        # convert to int durations for each element in the batch
        durations_int = torch.LongTensor(duration_preds.size(0), duration_preds.size(1)).zero_()  # (B, L_max)
        for line_idx in range(duration_preds.size(0)):
            end_prev, symbols_idx, durations_float = 0., [], []
            for symbol_id in range(duration_preds.size(1)):
                symb_dur = duration_preds[line_idx, symbol_id].item()
                if symb_dur != 0.:  # ignore 0 durations
                    symbols_idx.append(symbol_id)
                    durations_float.append([end_prev, end_prev + symb_dur])
                    end_prev += symb_dur
            int_durs = torch.LongTensor(duration_to_integer(durations_float, hparams))  # (L_max, )
            durations_int[line_idx, symbols_idx] = int_durs
        # put on GPU
        durations_int = durations_int.to(duration_preds.device, non_blocking=True).long()  # (B, L_max)

        return duration_preds, durations_int

    def pitch_shift(self, pitch_preds, pitch_factors, hparams, speaker_ids):
        ''' Pitch shift pitch predictions
            Pitch factors are assumed to be in Hz
        '''
        # keep track of unvoiced idx
        zero_idxs = (pitch_preds == 0.).nonzero()  # (N, 2)
        # pitch factors are F0 shifts in Hz
        for line_idx in range(pitch_preds.size(0)):
            speaker_id = speaker_ids[line_idx].item()
            pitch_mean = hparams.stats[f'spk {speaker_id}']['pitch']['mean']
            pitch_std = hparams.stats[f'spk {speaker_id}']['pitch']['std']
            pitch_preds[line_idx] = torch.exp(pitch_std * pitch_preds[line_idx] + pitch_mean)  # (L_max)
            # perform pitch shift in Hz domain
            pitch_preds[line_idx] += pitch_factors[line_idx]  # (L_max)
            # go back to log and re-normalize using pitch training stats
            pitch_preds[line_idx] = (torch.log(pitch_preds[line_idx]) - pitch_mean) / pitch_std  # (L_max)
        # set unvoiced idx to zero
        pitch_preds[zero_idxs[:, 0], zero_idxs[:, 1]] = 0.

        return pitch_preds

    def pitch_multiply(self, pitch_preds, pitch_factors):
        ''' Apply multiply transform to pitch prediction with respect to the mean

            Effects of factor values on the pitch:
                ]0, +inf[       amplify
                0               no effect
                ]-1, 0[         de-amplify
                -1              flatten
                ]-2, -1[        invert de-amplify
                -2              invert
                ]-inf, -2[      invert amplify
        '''
        # multiply pitch for each element in the batch
        for line_idx in range(pitch_preds.size(0)):
            # keep track of voiced and unvoiced idx
            non_zero_idxs = pitch_preds[line_idx].nonzero()  # (M, )
            zero_idxs = (pitch_preds[line_idx] == 0.).nonzero()  # (N, )
            # compute mean of voiced values
            mean_pitch = torch.mean(pitch_preds[line_idx, non_zero_idxs])
            # compute deviation to the mean for each pitch prediction
            pitch_deviation = pitch_preds[line_idx] - mean_pitch  # (L_max)
            # multiply factors to pitch deviation
            pitch_deviation *= pitch_factors[line_idx]  # (L_max)
            # add deviation to pitch predictions
            pitch_preds[line_idx] += pitch_deviation  # (L_max)
            # reset unvoiced values to 0
            pitch_preds[line_idx, zero_idxs] = 0.

        return pitch_preds

    def inference(self, inputs, pitch_transform, hparams, external_prosody=None,
                  external_embeddings=None, external_accent_emb=None):
        ''' Inference: synthesis uses external_accent_emb only (no refs).

        Args:
            inputs: (symbols, dur_factors, energy_factors, pitch_factors, input_lengths, speaker_ids)
            external_embeddings: (B, external_emb_dim) required
            external_accent_emb: (B, hidden_embed_dim) required for synthesis
        '''
        symbols, dur_factors, energy_factors, pitch_factors, input_lengths, speaker_ids = inputs

        if external_embeddings is None:
            raise ValueError("external_embeddings required for inference. Provide ECAPA speaker embedding.")
        spk_emb = torch.nn.functional.normalize(external_embeddings, p=2, dim=-1)
        spk_emb = self.spk_projection(spk_emb)

        if hasattr(self, 'accent_encoder') and self.accent_encoder is not None:
            if external_accent_emb is None:
                raise ValueError(
                    "external_accent_emb required for inference. "
                    "Provide --accent_emb_audios_dir or use a checkpoint with memorized_accent_emb (e.g. from adapt_accent)."
                )
            accent_emb = external_accent_emb
            combined_emb = accent_emb + spk_emb
            film_params_dict = self.style_adapter(combined_emb)
        else:
            film_params_dict = {
                'phoneme_encoder': None,
                'frame_decoder': None
            }

        # 4. Encode Phonemes (with accent FiLM conditioning)
        enc_outputs = self.phoneme_encoder(
            symbols,
            film_params_dict['phoneme_encoder'],
            input_lengths
        )  # (B, L_max, hidden_embed_dim)

        # 5. Get Prosody (Must be provided externally as predictor is removed)
        if external_prosody is None:
            raise ValueError("external_prosody must be provided for inference as the internal predictor has been removed.")

        duration_preds = external_prosody['duration_preds']
        durations_int = external_prosody['durations_int']
        energy_preds = external_prosody['energy_preds']
        pitch_preds = external_prosody['pitch_preds']

        # Apply factors
        duration_preds = duration_preds * dur_factors  # (B, L_max)
        duration_preds, durations_int = self.get_int_durations(duration_preds, hparams)  # (B, L_max)

        energy_preds = energy_preds * energy_factors
        # Enforce masking
        energy_preds[durations_int == 0] = 0.
        pitch_preds[durations_int == 0] = 0.

        if pitch_transform == 'add':
            pitch_preds = self.pitch_shift(pitch_preds, pitch_factors, hparams, speaker_ids)
        elif pitch_transform == 'multiply':
            pitch_preds = self.pitch_multiply(pitch_preds, pitch_factors)
        else:
            raise NotImplementedError

        # 6. Gaussian Upsampling (no FiLM, explicit prosody only)
        symbols_upsamp, weights = self.gaussian_upsampling(
            enc_outputs, duration_preds, durations_int, energy_preds, pitch_preds, input_lengths,
            film_params=None  # No FiLM on upsampling, like original
        )  # (B, T_max, hidden_embed_dim), (B, L_max, T_max)

        # Get sequence output lengths
        output_lengths = torch.sum(durations_int, dim=1)  # (B, )
        output_lengths = output_lengths.to(symbols_upsamp.device, non_blocking=True).long()
        output_lengths[output_lengths == 0] = 1  # Safety check

        assert(torch.max(output_lengths) == symbols_upsamp.size(1))

        # 7. Frame Decoder (with accent FiLM conditioning)
        mel_spec_preds = self.frame_decoder(
            symbols_upsamp,
            film_params_dict['frame_decoder'],
            output_lengths
        )  # (B, n_mel_channels, T_max)

        # Parse outputs
        encoder_preds = [duration_preds, durations_int, energy_preds, pitch_preds, input_lengths]
        decoder_preds = [mel_spec_preds, output_lengths]
        alignments = weights

        return encoder_preds, decoder_preds, alignments
