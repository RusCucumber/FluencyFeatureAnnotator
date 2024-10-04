from enum import Enum
from typing import List

from modules.common import RESOURCES_PATH
from modules.models.utils.base import DistilBertTokenClassifierBase

FINETUED_PUNCTUATION_PREDICTOR = RESOURCES_PATH / "pretrained_model/finetuned_punctuation_predictor_v1"

class PunctuationPredictor(DistilBertTokenClassifierBase):
    def __init__(self, n_class: int =2, device: str ="cpu") -> 'PunctuationPredictor':
        super().__init__(n_class=n_class, device=device)

    @classmethod
    def from_finetuned(cls, device: str ="cpu") -> 'PunctuationPredictor':
        return super().from_finetuned(FINETUED_PUNCTUATION_PREDICTOR, device=device)

    def preprocess_predict(self, text: str) -> List[Enum]:
        text = [text]
        return super().preprocess_predict(text)
