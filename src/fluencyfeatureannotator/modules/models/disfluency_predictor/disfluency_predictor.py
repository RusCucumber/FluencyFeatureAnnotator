from modules.common import RESOURCES_PATH
from modules.models.utils.base import DistilBertTokenClassifierBase, RobertaTokenClassifierBase

FINETUED_DISFLUENCY_PREDICTOR_BERT = RESOURCES_PATH / "pretrained_model/finetuned_disfluency_predictor_v1"
FINETUED_DISFLUENCY_PREDICTOR_ROBERTA = RESOURCES_PATH / "pretrained_model/finetuned_disfluency_predictor_v3"
FINETUED_DISFLUENCY_PREDICTOR_ROBERTA_L1 = RESOURCES_PATH / "pretrained_model/finetuned_disfluency_predictor_v4"

class DisfluencyPredictorBert(DistilBertTokenClassifierBase):
    def __init__(self, n_class: int =8, device: str ="cpu") -> 'DisfluencyPredictorBert':
        super().__init__(n_class=n_class, device=device)

    @classmethod
    def from_finetuned(cls, device: str ="cpu") -> 'DisfluencyPredictorBert':
        return super().from_finetuned(FINETUED_DISFLUENCY_PREDICTOR_BERT, device=device)

class DisfluencyPredictorRoberta(RobertaTokenClassifierBase):
    def __init__(self, n_class: int =2, ignore_tag: int =-100, device: str ="cpu") -> 'DisfluencyPredictorRoberta':
        super().__init__(n_class=n_class, ignore_tag=ignore_tag, device=device)

    @classmethod
    def from_finetuned(cls, device: str ="cpu", L1: bool =False) -> 'DisfluencyPredictorRoberta':
        if L1:
            return super().from_finetuned(FINETUED_DISFLUENCY_PREDICTOR_ROBERTA_L1, device=device)

        return super().from_finetuned(FINETUED_DISFLUENCY_PREDICTOR_ROBERTA, device=device)

