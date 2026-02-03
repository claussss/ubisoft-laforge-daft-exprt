import collections
import logging
import logging.handlers
import os
import random
import re
import time
import uuid

import librosa
import numpy as np
import torch

from scipy.io import wavfile
from shutil import rmtree

from daft_exprt.cleaners import collapse_whitespace, text_cleaner
from daft_exprt.extract_features import extract_energy, extract_pitch, mel_spectrogram_HiFi, rescale_wav_to_float32
from daft_exprt.symbols import ascii, eos, punctuation, whitespace
from daft_exprt.utils import chunker, launch_multi_process, plot_2d_data


_logger = logging.getLogger(__name__)
FILE_ROOT = os.path.dirname(os.path.realpath(__file__))


def phonemize_sentence(sentence, hparams, log_queue):
    ''' Phonemize sentence using MFA
    '''
    # get MFA variables
    dictionary = hparams.mfa_dictionary
    g2p_model = hparams.mfa_g2p_model
    # load dictionary and extract word transcriptions
    word_trans = collections.defaultdict(list)
    with open(dictionary, 'r', encoding='utf-8') as f:
        lines = [line.strip().split() for line in f.readlines()]
    for line in lines:
        word_trans[line[0].lower()].append(line[1:])
    # characters to consider in the sentence
    if hparams.language == 'english':
        all_chars = ascii + punctuation
    else:
        raise NotImplementedError()

    # clean sentence
    sentence = text_cleaner(sentence.strip(), hparams.language).lower().strip()
    # split sentence
    sent_words = re.findall(f"[\w']+|[{punctuation}]", sentence.lower().strip())
    # remove characters that are not letters or punctuation
    sent_words = [x for x in sent_words if len(re.sub(f'[^{all_chars}]', '', x)) != 0]
    # be sure to begin the sentence with a word and not a punctuation
    while sent_words[0] in punctuation:
        sent_words.pop(0)
    # keep only one punctuation type at the end
    punctuation_end = None
    while sent_words[-1] in punctuation:
        punctuation_end = sent_words.pop(-1)
    sent_words.append(punctuation_end)

    # phonemize words and add word boundaries
    sentence_phonemized, unk_words = [], []
    while len(sent_words) != 0:
        word = sent_words.pop(0)
        if word in word_trans:
            phones = random.choice(word_trans[word])
            sentence_phonemized.append(phones)
        else:
            unk_words.append(word)
            sentence_phonemized.append('<unk>')
        # at this point we pass to the next word
        # we must add a word boundary between two consecutive words
        if len(sent_words) != 0:
            word_bound = sent_words.pop(0) if sent_words[0] in punctuation else whitespace
            sentence_phonemized.append(word_bound)
    # add EOS token
    sentence_phonemized.append(eos)

    # use MFA g2p model to phonemize unknown words
    if len(unk_words) != 0:
        rand_name = str(uuid.uuid4())
        oovs = os.path.join(FILE_ROOT, f'{rand_name}_oovs.txt')
        with open(oovs, 'w', encoding='utf-8') as f:
            for word in unk_words:
                f.write(f'{word}\n')
        # generate transcription for unknown words
        oovs_trans = os.path.join(FILE_ROOT, f'{rand_name}_oovs_trans.txt')
        tmp_dir = os.path.join(FILE_ROOT, f'{rand_name}')
        os.system(f'mfa g2p {g2p_model} {oovs} {oovs_trans} -t {tmp_dir}')
        # extract transcriptions
        with open(oovs_trans, 'r', encoding='utf-8') as f:
            lines = [line.strip().split() for line in f.readlines()]
        for line in lines:
            transcription = line[1:]
            unk_idx = sentence_phonemized.index('<unk>')
            sentence_phonemized[unk_idx] = transcription
        # remove files
        os.remove(oovs)
        os.remove(oovs_trans)
        rmtree(tmp_dir, ignore_errors=True)

    return sentence_phonemized


def collate_tensors(batch_sentences, batch_dur_factors, batch_energy_factors,
                    batch_pitch_factors, pitch_transform,
                    batch_speaker_ids, batch_file_names, hparams,
                    external_prosody=None):
    ''' Collate tensors for synthesis. Synthesis uses external_accent_emb only; no ref .npz. '''
    batch = []
    for sentence, dur_factors, energy_factors, pitch_factors in \
        zip(batch_sentences, batch_dur_factors, batch_energy_factors, batch_pitch_factors):
        symbols = []
        for item in sentence:
            if isinstance(item, list):
                symbols += [hparams.symbols.index(phone) for phone in item]
            else:
                symbols.append(hparams.symbols.index(item))
        symbols = torch.IntTensor(symbols)
        if dur_factors is None:
            dur_factors = [1. for _ in range(len(symbols))]
        dur_factors = torch.FloatTensor(dur_factors)
        assert len(dur_factors) == len(symbols)
        if energy_factors is None:
            energy_factors = [1. for _ in range(len(symbols))]
        energy_factors = torch.FloatTensor(energy_factors)
        assert len(energy_factors) == len(symbols)
        if pitch_factors is None:
            if pitch_transform == 'add':
                pitch_factors = [0. for _ in range(len(symbols))]
            else:
                pitch_factors = [1. for _ in range(len(symbols))]
        pitch_factors = torch.FloatTensor(pitch_factors)
        assert len(pitch_factors) == len(symbols)
        batch.append([symbols, dur_factors, energy_factors, pitch_factors])

    input_lengths, ids_sorted_decreasing = \
        torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
    max_input_len = input_lengths[0]
    symbols = torch.LongTensor(len(batch), max_input_len).zero_()
    dur_factors = 1 + torch.FloatTensor(len(batch), max_input_len).zero_()
    energy_factors = 1 + torch.FloatTensor(len(batch), max_input_len).zero_()
    if pitch_transform == 'add':
        pitch_factors = torch.FloatTensor(len(batch), max_input_len).zero_()
    else:
        pitch_factors = 1 + torch.FloatTensor(len(batch), max_input_len).zero_()
    for i in range(len(ids_sorted_decreasing)):
        symbols[i, :batch[ids_sorted_decreasing[i]][0].size(0)] = batch[ids_sorted_decreasing[i]][0]
        dur_factors[i, :batch[ids_sorted_decreasing[i]][1].size(0)] = batch[ids_sorted_decreasing[i]][1]
        energy_factors[i, :batch[ids_sorted_decreasing[i]][2].size(0)] = batch[ids_sorted_decreasing[i]][2]
        pitch_factors[i, :batch[ids_sorted_decreasing[i]][3].size(0)] = batch[ids_sorted_decreasing[i]][3]

    file_names = []
    speaker_ids = torch.LongTensor(len(batch))
    for i in range(len(ids_sorted_decreasing)):
        file_names.append(batch_file_names[ids_sorted_decreasing[i]])
        speaker_ids[i] = batch_speaker_ids[ids_sorted_decreasing[i]]

    sorted_external = None
    if external_prosody is not None:
        assert len(external_prosody) == len(batch_sentences)
        sorted_external = [external_prosody[idx] for idx in ids_sorted_decreasing]
    return symbols, dur_factors, energy_factors, pitch_factors, input_lengths, speaker_ids, file_names, sorted_external


def _normalize_external_feature(values, zero_mask, target_stats, source_stats=None):
    """Apply source->target z-score mapping while preserving zeros."""
    values = values.clone()
    non_zero = (~zero_mask)
    if source_stats is not None:
        src_mean = source_stats['mean']
        src_std = source_stats['std']
        if src_std == 0:
            raise ValueError('Source stats std cannot be 0.')
        tmp = values[non_zero]
        tmp = (tmp - src_mean) / src_std
        tmp = tmp * target_stats['std'] + target_stats['mean']
        values[non_zero] = tmp
    tgt_std = target_stats['std']
    if tgt_std == 0:
        raise ValueError('Target speaker stats std cannot be 0.')
    tmp = values[non_zero]
    tmp = (tmp - target_stats['mean']) / tgt_std
    values[non_zero] = tmp
    values[zero_mask] = 0.
    return values


def generate_batch_mel_specs(model, batch_sentences, batch_dur_factors,
                             batch_energy_factors, batch_pitch_factors, pitch_transform,
                             batch_speaker_ids, batch_file_names, output_dir, hparams,
                             n_jobs, batch_external_prosody=None,
                             vocoder=None, source_stats=None,
                             alpha_dur=1.0, alpha_pitch=1.0, alpha_energy=1.0,
                             external_embeddings=None, external_accent_emb=None):
    ''' Generate batch mel-specs using Daft-Exprt. Synthesis uses external_accent_emb only; no refs. '''
    for idx, file_name in enumerate(batch_file_names):
        batch_file_names[idx] = file_name + f'_spk_{batch_speaker_ids[idx]}'
        _logger.info(f'Generating "{batch_sentences[idx]}" as "{batch_file_names[idx]}"')
    symbols, dur_factors, energy_factors, pitch_factors, input_lengths, speaker_ids, file_names, sorted_external_prosody = \
        collate_tensors(batch_sentences, batch_dur_factors, batch_energy_factors,
                        batch_pitch_factors, pitch_transform,
                        batch_speaker_ids, batch_file_names, hparams,
                        external_prosody=batch_external_prosody)
    gpu = next(model.parameters()).device
    speaker_ids_cpu = speaker_ids.clone()
    symbols = symbols.cuda(gpu, non_blocking=True).long()
    dur_factors = dur_factors.cuda(gpu, non_blocking=True).float()
    energy_factors = energy_factors.cuda(gpu, non_blocking=True).float()
    pitch_factors = pitch_factors.cuda(gpu, non_blocking=True).float()
    input_lengths = input_lengths.cuda(gpu, non_blocking=True).long()
    speaker_ids = speaker_ids_cpu.cuda(gpu, non_blocking=True).long()

    external_tensors = None
    if sorted_external_prosody is not None:
        max_len = symbols.size(1)
        batch_size = symbols.size(0)
        ext_duration = torch.FloatTensor(batch_size, max_len).zero_()
        ext_duration_int = torch.LongTensor(batch_size, max_len).zero_()
        ext_energy = torch.FloatTensor(batch_size, max_len).zero_()
        ext_pitch = torch.FloatTensor(batch_size, max_len).zero_()
        hop_in_seconds = hparams.hop_length / hparams.sampling_rate
        for idx, (entry, seq_len) in enumerate(zip(sorted_external_prosody, input_lengths.cpu().tolist())):
            assert len(entry['symbols']) == seq_len, \
                _logger.error(f'External prosody length mismatch for sample {file_names[idx]}')

            frames = torch.FloatTensor(entry['durations_frames'])

            # Apply variance exaggeration
            dur_mask = (frames > 0)
            if dur_mask.any() and alpha_dur != 1.0:
                dur_mean = frames[dur_mask].mean()
                frames[dur_mask] = dur_mean + alpha_dur * (frames[dur_mask] - dur_mean)
                frames = torch.clamp(frames, min=0.0)

            ext_duration[idx, :seq_len] = frames * hop_in_seconds
            ext_duration_int[idx, :seq_len] = torch.round(frames).long()

            energy_vals = torch.FloatTensor(entry['energy'])
            pitch_vals = torch.FloatTensor(entry['pitch'])
            energy_zero = (energy_vals == 0.)
            pitch_zero = (pitch_vals == 0.)
            speaker_id = speaker_ids_cpu[idx].item()
            spk_key = f'spk {speaker_id}'
            if spk_key not in hparams.stats and 'spk 0' in hparams.stats:
                spk_key = 'spk 0'
            if spk_key not in hparams.stats:
                raise KeyError(f"Speaker stats missing for 'spk {speaker_id}' and fallback 'spk 0' not in hparams.stats (keys: {list(hparams.stats.keys())})")
            energy_mean = hparams.stats[spk_key]['energy']['mean']
            energy_std = hparams.stats[spk_key]['energy']['std']
            pitch_mean = hparams.stats[spk_key]['pitch']['mean']
            pitch_std = hparams.stats[spk_key]['pitch']['std']
            if energy_std == 0 or pitch_std == 0:
                raise ValueError(f'Speaker stats not initialized for speaker ID {speaker_id}.')
            src_energy_stats = source_stats['energy'] if source_stats is not None else None
            src_pitch_stats = source_stats['pitch'] if source_stats is not None else None
            energy_vals = _normalize_external_feature(
                energy_vals, energy_zero,
                {'mean': energy_mean, 'std': energy_std},
                src_energy_stats)
            pitch_vals = _normalize_external_feature(
                pitch_vals, pitch_zero,
                {'mean': pitch_mean, 'std': pitch_std},
                src_pitch_stats)

            if alpha_energy != 1.0:
                energy_vals[~energy_zero] *= alpha_energy

            if alpha_pitch != 1.0:
                pitch_vals[~pitch_zero] *= alpha_pitch

            ext_energy[idx, :seq_len] = energy_vals
            ext_pitch[idx, :seq_len] = pitch_vals
        external_tensors = {
            'duration_preds': ext_duration.cuda(gpu, non_blocking=True).float(),
            'durations_int': ext_duration_int.cuda(gpu, non_blocking=True).long(),
            'energy_preds': ext_energy.cuda(gpu, non_blocking=True).float(),
            'pitch_preds': ext_pitch.cuda(gpu, non_blocking=True).float()
        }
    inputs = (symbols, dur_factors, energy_factors, pitch_factors, input_lengths, speaker_ids)
    encoder_preds, decoder_preds, alignments = model.inference(
        inputs, pitch_transform, hparams, external_tensors,
        external_embeddings=external_embeddings,
        external_accent_emb=external_accent_emb
    )
    # parse outputs
    duration_preds, durations_int, energy_preds, pitch_preds, input_lengths = encoder_preds
    mel_spec_preds, output_lengths = decoder_preds
    weights = alignments
    # transfer data to cpu and convert to numpy array
    duration_preds = duration_preds.detach().cpu().numpy()  # (B, L_max)
    durations_int = durations_int.detach().cpu().numpy()  # (B, L_max)
    energy_preds = energy_preds.detach().cpu().numpy()  # (B, L_max)
    pitch_preds = pitch_preds.detach().cpu().numpy()  # (B, L_max)
    input_lengths = input_lengths.detach().cpu().numpy()  # (B, )
    mel_spec_preds = mel_spec_preds.detach().cpu().numpy()  # (B, n_mel_channels, T_max)
    output_lengths = output_lengths.detach().cpu().numpy()  # (B)
    weights = weights.detach().cpu().numpy()  # (B, L_max, T_max)

    # save preds for each element in the batch
    predictions = {}
    for line_idx in range(mel_spec_preds.shape[0]):
        # crop prosody preds to the correct length
        duration_pred = duration_preds[line_idx, :input_lengths[line_idx]]  # (L, )
        duration_int = durations_int[line_idx, :input_lengths[line_idx]]  # (L, )
        energy_pred = energy_preds[line_idx, :input_lengths[line_idx]]  # (L, )
        pitch_pred = pitch_preds[line_idx, :input_lengths[line_idx]]  # (L, )
        # crop mel-spec to the correct length
        mel_spec_pred = mel_spec_preds[line_idx, :, :output_lengths[line_idx]]  # (n_mel_channels, T)
        # crop weights to the correct length
        weight = weights[line_idx, :input_lengths[line_idx], :output_lengths[line_idx]]
        # save generated spectrogram
        file_name = file_names[line_idx]
        np.savez(os.path.join(output_dir, f'{file_name}.npz'), mel_spec=mel_spec_pred)
        # store predictions
        predictions[f'{file_name}'] = [duration_pred, duration_int, energy_pred, pitch_pred, mel_spec_pred, weight]

    if vocoder is None:
        raise ValueError(
            "HiFi-GAN vocoder required for mel-to-wave. "
            "Provide --vocoder_checkpoint in synthesize.py or load a vocoder before calling generate."
        )
    for file_name, (_, _, _, _, mel_spec, weight) in predictions.items():
        plot_2d_data(data=(mel_spec, weight),
                     x_labels=('Mel-Spec Prediction', 'Alignments'),
                     filename=os.path.join(output_dir, file_name + '.png'))
        audio = vocoder.infer(mel_spec)
        audio_int16 = (audio * 32767.5).clip(min=-32768, max=32767).astype(np.int16)
        wavfile.write(os.path.join(output_dir, f'{file_name}.wav'), hparams.sampling_rate, audio_int16)

    return predictions


def generate_mel_specs(model, sentences, file_names, speaker_ids, output_dir, hparams,
                       dur_factors=None, energy_factors=None, pitch_factors=None, batch_size=1,
                       n_jobs=1, get_time_perf=False, external_prosody=None,
                       vocoder=None, source_stats=None,
                       alpha_dur=None, alpha_pitch=None, alpha_energy=None,
                       external_embeddings=None, external_accent_emb=None):
    ''' Generate mel-specs using Daft-Exprt. Synthesis uses external_accent_emb only; no refs. '''
    dur_factors = [None for _ in range(len(sentences))] if dur_factors is None else dur_factors
    energy_factors = [None for _ in range(len(sentences))] if energy_factors is None else energy_factors
    pitch_factors = ['add', [None for _ in range(len(sentences))]] if pitch_factors is None else pitch_factors
    pitch_transform = pitch_factors[0].lower()
    pitch_factors = pitch_factors[1]
    assert pitch_transform in ['add', 'multiply']
    assert len(file_names) == len(sentences)
    assert len(speaker_ids) == len(sentences)
    assert len(dur_factors) == len(sentences)
    assert len(energy_factors) == len(sentences)
    assert len(pitch_factors) == len(sentences)
    if external_prosody is not None:
        assert len(external_prosody) == len(sentences)

    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    predictions, time_per_batch = {}, []
    sentence_chunks = list(chunker(sentences, batch_size))
    dur_chunks = list(chunker(dur_factors, batch_size))
    energy_chunks = list(chunker(energy_factors, batch_size))
    pitch_chunks = list(chunker(pitch_factors, batch_size))
    speaker_chunks = list(chunker(speaker_ids, batch_size))
    file_chunks = list(chunker(file_names, batch_size))
    external_chunks = list(chunker(external_prosody, batch_size)) if external_prosody is not None else [None] * len(sentence_chunks)
    external_embeddings_chunks = list(chunker(external_embeddings, batch_size)) if external_embeddings is not None else [None] * len(sentence_chunks)
    external_accent_chunks = list(chunker(external_accent_emb, batch_size)) if external_accent_emb is not None else [None] * len(sentence_chunks)

    with torch.no_grad():
        for idx in range(len(sentence_chunks)):
            batch_external = external_chunks[idx]
            batch_external_embeddings = external_embeddings_chunks[idx]
            batch_accent_emb = external_accent_chunks[idx]
            sentence_begin = time.time() if get_time_perf else None
            batch_predictions = generate_batch_mel_specs(
                model, sentence_chunks[idx], dur_chunks[idx], energy_chunks[idx], pitch_chunks[idx],
                pitch_transform, speaker_chunks[idx], file_chunks[idx], output_dir, hparams,
                n_jobs, batch_external, vocoder,
                source_stats=source_stats,
                alpha_dur=alpha_dur, alpha_pitch=alpha_pitch, alpha_energy=alpha_energy,
                external_embeddings=batch_external_embeddings, external_accent_emb=batch_accent_emb)
            predictions.update(batch_predictions)
            time_per_batch += [time.time() - sentence_begin] if get_time_perf else []

    # display overall time performance
    if get_time_perf:
        # get duration of each sentence
        durations = []
        for prediction in predictions.values():
            _, _, _, _, mel_spec, _ = prediction
            nb_frames = mel_spec.shape[1]
            nb_wav_samples = (nb_frames - 1) * hparams.hop_length + hparams.filter_length
            if hparams.centered:
                nb_wav_samples -= 2 * int(hparams.filter_length / 2)
            duration = nb_wav_samples / hparams.sampling_rate
            durations.append(duration)
        _logger.info(f'')
        _logger.info(f'{len(predictions)} sentences ({sum(durations):.2f}s) generated in {sum(time_per_batch):.2f}s')
        _logger.info(f'DaftExprt RTF: {sum(durations)/sum(time_per_batch):.2f}')

    return predictions


def extract_reference_parameters(audio_ref, output_dir, hparams, ref_name=None):
    ''' Extract energy, pitch and mel-spectrogram parameters from audio
        Save numpy arrays to .npz file
    '''
    # check if file name already exists
    os.makedirs(output_dir, exist_ok=True)
    file_name = ref_name if ref_name is not None else os.path.basename(audio_ref).replace('.wav', '')
    ref_file = os.path.join(output_dir, f'{file_name}.npz')
    if not os.path.isfile(ref_file):
        # read wav file to range [-1, 1] in np.float32
        wav, fs = librosa.load(audio_ref, sr=hparams.sampling_rate)
        wav = rescale_wav_to_float32(wav)
        # get log pitch
        pitch = extract_pitch(wav, fs, hparams)
        # extract mel-spectrogram
        mel_spec = mel_spectrogram_HiFi(wav, hparams)
        # get energy
        energy = extract_energy(np.exp(mel_spec))
        # check sizes are correct
        # Fix potential 1-frame mismatch between pitch/energy and mel-spec
        min_len = min(len(pitch), len(energy), mel_spec.shape[1])
        if len(pitch) > min_len:
            pitch = pitch[:min_len]
        if len(energy) > min_len:
            energy = energy[:min_len]
        if mel_spec.shape[1] > min_len:
            mel_spec = mel_spec[:, :min_len]

        assert(len(pitch) == mel_spec.shape[1]), f'{len(pitch)} -- {mel_spec.shape[1]}'
        assert(len(energy) == mel_spec.shape[1]), f'{len(energy)} -- {mel_spec.shape[1]}'
        # save references to .npz file
        np.savez(ref_file, energy=energy, pitch=pitch, mel_spec=mel_spec)


def prepare_sentences_for_inference(text_file, output_dir, hparams, n_jobs):
    ''' Phonemize and format sentences to synthesize
    '''
    # create output directory or delete everything if it already exists
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)

    # extract sentences to synthesize
    assert(os.path.isfile(text_file)), _logger.error(f'There is no such file {text_file}')
    with open(os.path.join(text_file), 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    file_names = [f'{os.path.basename(text_file)}_line{idx}' for idx in range(len(sentences))]
    # phonemize
    hparams.update_mfa_paths()
    sentences = launch_multi_process(iterable=sentences, func=phonemize_sentence,
                                     n_jobs=n_jobs, timer_verbose=False, hparams=hparams)

    # save the sentences in a file
    with open(os.path.join(output_dir, 'sentences_to_generate.txt'), 'w', encoding='utf-8') as f:
        for sentence, file_name in zip(sentences, file_names):
            text = ''
            for item in sentence:
                if isinstance(item, list):  # corresponds to phonemes of a word
                    item = '{' + ' '.join(item) + '}'
                text = f'{text} {item} '
            text = collapse_whitespace(text).strip()
            f.write(f'{file_name}|{text}\n')

    return sentences, file_names
