import json
import math
from functools import partial
from typing import List

import torch
import torchaudio

from .data_module import LibriSpeechDataModule
from .lightning import Batch

_decibel = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
_gain = pow(10, 0.05 * _decibel)

_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)
_speed_perturb_transform = torchaudio.transforms.SpeedPerturbation(orig_freq=16000, factors=[0.9, 1.0, 1.1])


def _piecewise_linear_log(x):
    x = x * _gain
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class GlobalStatsNormalization(torch.nn.Module):
    def __init__(self, global_stats_path):
        super().__init__()

        with open(global_stats_path) as f:
            blob = json.loads(f.read())

        self.mean = torch.tensor(blob["mean"])
        self.invstddev = torch.tensor(blob["invstddev"])

    def forward(self, input):
        return (input - self.mean) * self.invstddev


def _extract_labels(tokenizer, samples: List):
    targets = [tokenizer.encode_flatten(sample[2].lower()) for sample in samples]
    lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
    targets = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(elem) for elem in targets],
        batch_first=True,
        padding_value=1.0,
    ).to(dtype=torch.int32)
    return targets, lengths


def _extract_features(data_pipeline, samples: List):
    samples = [_speed_perturb_transform(sample[0].squeeze()) for sample in samples]
    mel_features = [_spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
    features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
    features = data_pipeline(features)
    lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
    return features, lengths


class TrainTransform:
    def __init__(self, global_stats_path: str, tokenizer):
        self.tokenizer = tokenizer
        self.train_data_pipeline = torch.nn.Sequential(
            FunctionalModule(_piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.TimeMasking(100, p=0.2),
            torchaudio.transforms.TimeMasking(100, p=0.2),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        )

    def __call__(self, samples: List):
        features, feature_lengths = _extract_features(self.train_data_pipeline, samples)
        targets, target_lengths = _extract_labels(self.tokenizer, samples)
        return Batch(features, feature_lengths, targets, target_lengths)


class ValTransform:
    def __init__(self, global_stats_path: str, tokenizer):
        self.tokenizer = tokenizer
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(_piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
        )

    def __call__(self, samples: List):
        features, feature_lengths = _extract_features(self.valid_data_pipeline, samples)
        targets, target_lengths = _extract_labels(self.tokenizer, samples)
        return Batch(features, feature_lengths, targets, target_lengths)


class TestTransform:
    def __init__(self, global_stats_path: str, tokenizer):
        self.val_transforms = ValTransform(global_stats_path, tokenizer)

    def __call__(self, sample):
        return self.val_transforms([sample]), [sample]


def get_data_module(librispeech_path, global_stats_path, tokenizer):
    train_transform = TrainTransform(global_stats_path=global_stats_path, tokenizer=tokenizer)
    val_transform = ValTransform(global_stats_path=global_stats_path, tokenizer=tokenizer)
    test_transform = TestTransform(global_stats_path=global_stats_path, tokenizer=tokenizer)
    return LibriSpeechDataModule(
        librispeech_path=librispeech_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        max_tokens=2000,
        batch_size=None,
    )
