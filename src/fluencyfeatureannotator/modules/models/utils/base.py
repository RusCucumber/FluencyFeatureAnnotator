import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from modules.models.utils.preprocess import DistilBertPreprocess, RobertaPreprocess
from torch import nn
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import DistilBertForTokenClassification, RobertaModel


class DistilBertTokenClassifierBase:
    def __init__(self, n_class: int =2, device: str ="cpu") -> 'DistilBertTokenClassifierBase':
        self._model = DistilBertForTokenClassification.from_pretrained("distilbert-base-cased", num_labels=n_class)
        self._model.to(device)
        self._tag = None
        self.__tagging = None
        self.preprocess = DistilBertPreprocess(device=device)

    @property
    def model(self) -> DistilBertForTokenClassification:
        return self._model

    @property
    def tag(self) -> List[Enum]:
        return self._tag

    @tag.setter
    def tag(self, tag_dict: Dict[str, int]) -> None:
        self._tag = Enum("Tag", tag_dict)
        self.__tagging = np.frompyfunc(self._tag, 1, 1)

    @classmethod
    def from_finetuned(cls, model_dir: Union[str, Path], device: str ="cpu") -> 'DistilBertTokenClassifierBase':
        if isinstance(model_dir, str):
            model_dir = Path(model_dir).resolve()

        with open(model_dir / "config.json") as f:
            config = json.load(f)

        classifier = cls(n_class=config["n_class"], device=device)
        classifier.model.load_state_dict(
            torch.load(model_dir / "finetuned_model.pt", map_location=torch.device(device))
        )
        classifier.tag = config["tag"]

        return classifier

    def __call__(self, text: str) -> List[Enum]:
        return self.preprocess_predict(text)

    def tagging(self, prediction: np.ndarray) -> np.ndarray:
        if self._tag is None:
            raise RuntimeError("tag is not defined")

        return self.__tagging(prediction)

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, masks: torch.Tensor) -> np.ndarray:
        self._model.eval()

        logits = self._model(input_ids, attention_mask=attention_mask)["logits"]
        outputs = softmax(logits, dim=2).to("cpu")

        preds = np.argmax(outputs.detach().numpy(), axis=2)

        predictions = []
        for p, m in zip(preds, masks):
            tagged_pred = self.tagging(p[m])
            predictions.append(tagged_pred)

        return predictions

    def preprocess_predict(self, text: str) -> np.ndarray:
        args = self.preprocess(text)
        return self.predict(*args)


class RobertaBiLstm(nn.Module):
    def __init__(
            self,
            n_class: int =2,
            freeze_roberta: bool =False,
            ignore_tag: int =-100,
            device: str ="cpu"
    ) -> None:
        super().__init__()

        self.roberta = RobertaModel.from_pretrained("FacebookAI/roberta-base")
        self.bilstm = nn.LSTM(
            input_size=self.roberta.config.hidden_size,
            hidden_size=self.roberta.config.hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.linear = nn.Linear(self.roberta.config.hidden_size * 2, n_class)
        self.classifer = nn.LogSoftmax(dim=2)
        self.criterion = nn.NLLLoss(ignore_index=ignore_tag)

        if freeze_roberta:
            self.freeze_roberta()

        self.ignore_tag = ignore_tag
        self.n_class = n_class

        self.roberta.to(device)
        self.bilstm.to(device)
        self.linear.to(device)
        self.classifer.to(device)
        self.criterion.to(device)

    def freeze_roberta(self) -> None:
        for params in self.roberta.parameters():
            params.requires_grad = False

    def get_input_len(self, attention_mask: torch.Tensor) -> torch.Tensor:
        input_len = attention_mask.sum(axis=1, dtype=torch.int64)
        return input_len.to("cpu")

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            tags: torch.Tensor,
            return_logit: bool =False
        ) -> torch.Tensor:
        h = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = h.last_hidden_state
        input_len = self.get_input_len(attention_mask)

        packed = pack_padded_sequence(
            last_hidden_state, input_len,
            batch_first=True, enforce_sorted=False
        )
        h, _ = self.bilstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0)

        h = self.linear(h)
        logit = self.classifer(h)
        if return_logit:
            return logit

        loss = self.criterion(logit.view(-1, self.n_class), tags[:, :input_len.max()].clone().view(-1))

        return loss

    def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            tags: torch.Tensor #TODO: offset mapping から mask を取得する処理を追加
    ) -> list:
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, tags, return_logit=True)
            out = logits.argmax(dim=2)

            input_len = self.get_input_len(attention_mask)
            pred = []
            for idx, tag in enumerate(tags[:, :input_len.max()]):
                tag_mask = (tag != self.ignore_tag)
                pred.append(out[idx, tag_mask].detach().clone().tolist())

            return pred


class RobertaSequenceClassifier(nn.Module):
    def __init__(
            self,
            n_class: int =2,
            freeze_roberta: bool =False,
            ignore_tag: int =-100,
            device: str ="cpu"
    ) -> None:
        super().__init__()

        self.roberta = RobertaModel.from_pretrained("FacebookAI/roberta-base")
        self.linear = nn.Linear(self.roberta.config.hidden_size, n_class)
        self.classifer = nn.LogSoftmax(dim=2)
        self.criterion = nn.NLLLoss(ignore_index=ignore_tag)

        if freeze_roberta:
            self.freeze_roberta()

        self.ignore_tag = ignore_tag
        self.n_class = n_class

        self.roberta.to(device)
        self.linear.to(device)
        self.classifer.to(device)
        self.criterion.to(device)

    def freeze_roberta(self) -> None:
        for params in self.roberta.parameters():
            params.requires_grad = False

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            tags: torch.Tensor,
            return_logit: bool =False
        ) -> torch.Tensor:
        h = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = h.last_hidden_state

        h = self.linear(last_hidden_state)
        logit = self.classifer(h)
        if return_logit:
            return logit

        loss = self.criterion(logit.view(-1, self.n_class), tags.view(-1))

        return loss

    def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            tags: torch.Tensor #TODO: offset mapping から mask を取得する処理を追加
    ) -> list:
        with torch.no_grad():
            z = self.forward(input_ids, attention_mask, tags, return_logit=True)
            out = z.argmax(dim=2)

            pred = []
            for idx, tag in enumerate(tags):
                tag_mask = (tag != self.ignore_tag)
                pred.append(out[idx, tag_mask].detach().clone().tolist())

            return pred

class RobertaTokenClassifierBase:
    def __init__(
            self,
            n_class: int =2,
            ignore_tag: int =-100,
            device: str ="cpu"
    ) -> None:
        self._model = nn.DataParallel(RobertaSequenceClassifier(n_class, ignore_tag=ignore_tag, device=device))
        self._tag = None
        self.__tagging = None
        self.preprocess = RobertaPreprocess(device=device)

    @property
    def model(self):
        return self._model

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, tag_dict: dict):
        self._tag = Enum("Tag", tag_dict)
        self.__tagging = np.frompyfunc(self._tag, nin=1, nout=1)

    @classmethod
    def from_finetuned(cls, model_dir: Path, device: str ="cpu"):
        if isinstance(model_dir, str):
            model_dir = Path(model_dir).resolve()

        with open(model_dir / "config.json") as f:
            config = json.load(f)

        classifier = cls(n_class=config["n_class"], ignore_tag=config["ignore_tag"], device=device)
        classifier.model.load_state_dict(
            torch.load(model_dir / "checkpoint.pt", map_location=torch.device(device))
        )
        classifier.tag = config["tag"]

        return classifier

    def __call__(self, text: List[Union[str, List[str]]]) -> List[np.ndarray]:
        return self.preprocess_predict(text)

    def tagging(self, prediction: np.ndarray) -> np.ndarray:
        if self._tag is None:
            raise RuntimeError("tag is not defined")

        return self.__tagging(prediction)

    def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            tags: torch.Tensor
    ) -> List[np.ndarray]:
        self._model.eval()
        pred_raw = self.model.module.predict(input_ids, attention_mask, tags)

        predictions = []
        for p in pred_raw:
            if isinstance(p, list):
                p = np.array(p)

            tagged_pred = self.tagging(p)
            predictions.append(tagged_pred)

        return predictions

    def preprocess_predict(self, text: List[Union[str, List[str]]]) -> List[np.ndarray]:
        input_ids, attention_mask, tags = self.preprocess(text)
        return self.predict(input_ids, attention_mask, tags)

