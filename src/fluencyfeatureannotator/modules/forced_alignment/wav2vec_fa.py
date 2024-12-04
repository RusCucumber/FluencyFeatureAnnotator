"""
1. 音声・テキストデータの読み込み
2. テキストデータの前処理
3. rev 形式への変換
4. json ファイルとして保存

※ バッチ処理はひとまず考えないでOK
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F

CWD = Path(__file__).parent
sys.path.append(
    str(CWD / "utils")
)

from text.cleaners import english_cleaners

PRETRAINED_MODEL_PATH = CWD / Path("checkpoint/epoch=9-step=65369.ckpt")
NORMALIZATION_PARAMETER_PATH = CWD / Path("resources/global_stats.json")

PUNCTUATIONS = [
    ".", ",", "!", "?", "[", "]", "(", ")",
    "{", "}", "'", "\"", "/", "<", ">"
]

TARGET_CHANNEL = 0


class Wav2VecCTCFA:
    def __init__(
        self,
        device: str ="cpu"
    ):
        self.bundle = torchaudio.pipelines.MMS_FA
        self.model = self.bundle.get_model(with_star=True).to(device)
        self.device = device

    def load_audio(self, audio_file_path: Path) -> Tuple[torch.tensor, int]:
        waveform, sample_rate = torchaudio.load(str(audio_file_path))

        return waveform, sample_rate

    def resample(self, waveform: torch.tensor, sample_rate: int) -> torch.tensor:
        transformed_waveform = waveform[TARGET_CHANNEL, :].view(1, -1)

        target_sample_rate = self.bundle.sample_rate
        if sample_rate == target_sample_rate:
            return transformed_waveform

        transformed_waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(transformed_waveform)
        return transformed_waveform

    def preprocess_text(self, text: str) -> str:
        percents_pattern = r"(\d{2})%"
        cleaned_text = re.sub(
            percents_pattern, r"\1 percents", text
        )

        if "1%" in cleaned_text:
            cleaned_text = cleaned_text.replace("1%", "1 percent")
        if "%" in cleaned_text:
            cleaned_text = cleaned_text.replace("%", " percents")

        cleaned_text = english_cleaners(cleaned_text)

        for punctuation in PUNCTUATIONS:
            cleaned_text = cleaned_text.replace(punctuation, "")

        and_pattern = r"(\w)&(\w)"
        cleaned_text = re.sub(
            and_pattern, r"\1 & \2", cleaned_text
        )

        if "&" in cleaned_text:
            cleaned_text = cleaned_text.replace("&", "and")

        return cleaned_text

    def tokenize(self, transcript: str) -> List[str]:
        dictionary = self.bundle.get_dict()

        tokenized_transcript = []
        for word in transcript.split():
            for char in word:
                tokenized_transcript.append(dictionary[char])

        return tokenized_transcript

    def preprocess(self, audio_file_path: Union[str, Path], text: str) -> Tuple[torch.Tensor, List[str], int, str]:
        waveform, sample_rate = self.load_audio(audio_file_path)
        waveform = self.resample(waveform, sample_rate)

        cleaned_text = self.preprocess_text(text)
        tokens = self.tokenize(cleaned_text)

        return waveform, tokens, cleaned_text

    def frame_2_sec(self, frame: int, ratio: float, sample_rate: int) -> float:
        return int(frame * ratio) / sample_rate

    def to_dataframe(self, token_spans: list, transcript: str, ratio: float, sample_rate: int) -> pd.DataFrame:
        lengths = [len(word) for word in transcript.split()]
        if len(token_spans) != sum(lengths):
            raise RuntimeError(f"N tokens is not equal: {len(token_spans)} != {sum(lengths)}")

        i = 0
        data = []
        for l, chars in zip(lengths, transcript.split()):
            start_time = token_spans[i].start
            end_time = token_spans[i + l - 1].end

            start_time = self.frame_2_sec(start_time, ratio, sample_rate)
            end_time = self.frame_2_sec(end_time, ratio, sample_rate)

            row = [chars, start_time, end_time]
            data.append(row)

            i += l

        df_timestamp = pd.DataFrame(data, columns=["word", "start_time", "end_time"])

        return df_timestamp

    def align(self, audio_file_path: Path, text: str) -> Tuple[List[list], float]:
        waveform, tokens, cleaned_text = self.preprocess(audio_file_path, text)

        with torch.inference_mode():
            emission, _ = self.model(waveform.to(self.device))

        targets = torch.tensor([tokens], dtype=torch.int32, device=self.device)
        alignments, scores = F.forced_align(emission, targets, blank=0)

        alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
        scores = scores.exp()  # convert back to probability

        token_spans = F.merge_tokens(alignments, scores)

        ratio = waveform.size(1) / emission.size(1)

        df_timestamp = self.to_dataframe(token_spans, cleaned_text, ratio, self.bundle.sample_rate)

        return df_timestamp

