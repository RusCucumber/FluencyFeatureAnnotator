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

from .lightning import AcousticModelModule
from .tokenizer import EnglishPhonemeTokenizer
from .transforms import TestTransform

CWD = Path(__file__).parent
sys.path.append(
    str(CWD / "utils")
)

from text.cleaners import english_cleaners
from visualize import plot_alignments, plot_scores

PRETRAINED_MODEL_PATH = CWD / Path("checkpoint/epoch=9-step=65369.ckpt")
NORMALIZATION_PARAMETER_PATH = CWD / Path("resources/global_stats.json")

PUNCTUATIONS = [
    ".", ",", "!", "?", "[", "]", "(", ")",
    "{", "}", "'", "\"", "/", "<", ">"
]

TARGET_CHANNEL = 0

class LessPeakyCTCFA:
    def __init__(
        self,
        normalization_param_path: Union[Path, str] = NORMALIZATION_PARAMETER_PATH,
        pretrained_model_path: Union[Path, str] = PRETRAINED_MODEL_PATH,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = EnglishPhonemeTokenizer()

        global_stat_path = normalization_param_path
        self.transform = TestTransform(
            global_stats_path=global_stat_path,
            tokenizer=self.tokenizer
        )

        self.model = AcousticModelModule.load_from_checkpoint(
            pretrained_model_path,
            tokenizer=self.tokenizer
        ).to(self.device).eval()

    def unflatten(self, list_: list, lengths: List[int]) -> List[list]:
        assert len(list_) == sum(lengths)
        i = 0
        ret = []
        for l in lengths:
            ret.append(list_[i : i + l])
            i += l
        return ret

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

    def preprocess(
        self,
        audio_file_path: Union[str, Path],
        text: str
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], str, int]:
        cleaned_text = self.preprocess_text(text)

        waveform, sample_rate = torchaudio.load(audio_file_path)
        speaker_id, chapter_id, utterance_id = 0, 0, 0

        sample = (waveform, sample_rate, cleaned_text, speaker_id, chapter_id, utterance_id)
        batch, _ = self.transform(sample)

        tokenized_text = self.tokenizer.encode_flatten(cleaned_text)

        return batch, waveform, tokenized_text, cleaned_text, sample_rate

    def forward(
        self,
        batch: torch.Tensor,
        tokenized_text: torch.Tensor,
        cleand_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor, List[list]]:
        emission = self.model.forward(batch)

        aligned_tokens, alignment_score = self.model.align(
            batch,
            tokenized_text
        )

        token_spans = torchaudio.functional.merge_tokens(
            aligned_tokens, alignment_score
        )

        word_spans = self.unflatten(
            token_spans,
            [len(self.tokenizer.encode_flatten(word)) for word in cleand_text.split()]
        )

        return emission, alignment_score, word_spans

    def to_dataframe(
        self,
        waveform: torch.Tensor,
        emission: torch.Tensor,
        word_spans: List[list],
        cleaned_text: str,
        sample_rate: int
    ) -> pd.DataFrame:
        if self.device == "cuda":
            waveform = waveform.to("cpu")
            emission = emission.to("cpu")

        n_frames = emission.size(1)
        ratio = waveform.size(1) / n_frames

        words = cleaned_text.split()

        data = []
        columns = ["word", "start_time", "end_time"]
        for spans, word in zip(word_spans, words):
            start_time = int(ratio * spans[0].start) / sample_rate
            end_time = int(ratio * spans[-1].end) / sample_rate

            data.append([word, start_time, end_time])

        df_fa = pd.DataFrame(data, columns=columns)

        return df_fa

    def align(
        self,
        audio_file_path: Union[str, Path],
        text: str,
        visualize_result: bool =False
    ) -> pd.DataFrame:
        batch, waveform, tokenized_text, cleaned_text, sample_rate \
            = self.preprocess(audio_file_path, text)

        emission, alignment_score, word_spans = self.forward(
            batch, tokenized_text, cleaned_text
        )

        df_fa = self.to_dataframe(
            waveform, emission, word_spans, cleaned_text, sample_rate
        )

        if visualize_result:
            plot_alignments(
                waveform, word_spans, emission, cleaned_text.split()
            )
            plot_scores(
                word_spans, alignment_score, self.tokenizer
            )

        return df_fa

