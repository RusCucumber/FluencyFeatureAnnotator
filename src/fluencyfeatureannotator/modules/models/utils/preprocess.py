from typing import Dict, Tuple

import numpy as np
import torch
from torch import tensor
from transformers import DistilBertTokenizerFast, RobertaTokenizerFast


class DistilBertPreprocess:
    def __init__(self, device: str ="cpu") -> 'DistilBertPreprocess':
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")
        self._tokenize_config = {
            "is_split_into_words": True,
            "return_offsets_mapping": True,
            "padding": True,
            "truncation": True
        }
        self.__device = device

    @property
    def tokenize_config(self) -> Dict[str, bool]:
        return self._tokenize_config

    @tokenize_config.setter
    def tokenize_config(self, key: str, val: bool) -> None:
        self._tokenize_config[key] = val

    def generate_mask(self, offset_mapping: torch.Tensor) -> torch.Tensor:
        if isinstance(offset_mapping, torch.Tensor):
            offset_mapping = offset_mapping.to("cpu").detach().numpy()
        elif isinstance(offset_mapping, list):
            offset_mapping = np.array(offset_mapping)

        left_mask = (offset_mapping[:,:,0] == 0)
        right_mask = (offset_mapping[:,:,1] != 0)
        return left_mask & right_mask

    def tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_encodings = self.tokenizer(text, **self._tokenize_config)

        input_ids = tensor(text_encodings["input_ids"], device=self.__device)
        attention_mask = tensor(text_encodings["attention_mask"], device=self.__device)
        offset_mapping = text_encodings["offset_mapping"]

        return input_ids, attention_mask, offset_mapping

    def __call__(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids, attention_mask, offset_mapping = self.tokenize(text)
        mask = self.generate_mask(offset_mapping)

        return input_ids, attention_mask, mask

class RobertaPreprocess:
    def __init__(self, device: str ="cpu") -> 'RobertaPreprocess':
        self.tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
        self.tokenizer.add_prefix_space = True

        self._tokenize_config = {
            "is_split_into_words": True,
            "return_offsets_mapping": True,
            "padding": True,
            "return_tensors": "pt"
        }
        self.__device = device

    @property
    def tokenize_config(self) -> Dict[str, bool]:
        return self._tokenize_config

    @tokenize_config.setter
    def tokenize_config(self, key: str, val: bool) -> None:
        self._tokenize_config[key] = val

    def generate_ignore_tags(self, offset_mapping: torch.Tensor) -> torch.Tensor:
        n_data, n_seq, _ = offset_mapping.shape
        tags = torch.full((n_data, n_seq), -100, dtype=torch.int64)

        left_mask = (offset_mapping[:, :, 0] == 0)
        right_mask = (offset_mapping[:, :, 1] != 0)
        mask = left_mask & right_mask

        tags[mask] = 0

        tags = tags.to(self.__device)

        return tags

    def tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_encodings = self.tokenizer(text, **self._tokenize_config)

        input_ids = text_encodings["input_ids"]
        attention_mask = text_encodings["attention_mask"]
        offset_mapping = text_encodings["offset_mapping"]

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(self.__device)
            attention_mask = attention_mask.to(self.__device)
        else:
            input_ids = torch.tensor(input_ids, device=self.__device)
            attention_mask = torch.tensor(attention_mask, device=self.__device)


        return input_ids, attention_mask, offset_mapping

    def __call__(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids, attention_mask, offset_mapping = self.tokenize(text)
        tags = self.generate_ignore_tags(offset_mapping)

        return input_ids, attention_mask, tags

