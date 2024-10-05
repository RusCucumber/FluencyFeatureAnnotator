from typing import List, Tuple

from modules.common import DisfluencyEnum, Turn
from modules.models import DisfluencyPredictorBert, DisfluencyPredictorRoberta
from modules.pipeline.utils.base import ModuleBase


class Pruning(ModuleBase):
    def __init__(self, device: str ="cpu", model: str ="roberta") -> 'Pruning':
        super().__init__()
        if model == "roberta":
            print("RoBERTa model was selected!")
            self._model = DisfluencyPredictorRoberta.from_finetuned(device=device)
        elif model == "roberta_L1":
            print("RoBERTa (finetuned by only L1 dataset) model was selected!")
            self._model = DisfluencyPredictorRoberta.from_finetuned(device=device, L1=True)
        elif model == "bert":
            print("BERT model was selected!")
            self._model = DisfluencyPredictorBert.from_finetuned(device=device)
        else:
            raise RuntimeError(f"model {model} is not found. You should choose a model from \"roberta\" or \"bert\"")

    def predict(self, turn: Turn) -> Turn:
        turn, d_tags = self.predict_disfluency(turn)
        turn = self.pruning(turn, d_tags)

        return turn

    def predict_disfluency(self, turn: Turn) -> Tuple[Turn, List[DisfluencyEnum]]:
        self._turn_type_checker(turn)

        text = []
        sentence = []
        for sent in turn.clauses:
            for word in sent.words:
                sentence.append(word.text)
            text.append(sentence)
            sentence = []

        d_tags = self._model(text)

        return turn, d_tags

    def pruning(self, turn: Turn, disfluency_tags: List[DisfluencyEnum]) -> Turn:
        self._turn_type_checker(turn)

        prev_last_wid = 0
        for sentence, d_tag in zip(turn.clauses, disfluency_tags):
            if len(sentence.words) == 0:
                continue

            for indices, disfluency_list in self.__annotation_generator(sentence, d_tag):
                sentence.annotate_disfluency(indices, disfluency_list)

            sentence.ignore_disfluency()

            # 1文に含まれる単語全てが disfluency と判定された場合
            # -> prev_last_wid は一つ前の clause の最後の idx + 1 なので，次の clause に移れば良い
            if len(sentence.words) == 0:
                continue

            if sentence.words[0].idx != prev_last_wid:
                sentence.reset_index(idx=prev_last_wid)

            prev_last_wid = sentence.words[-1].idx + 1

        turn.reset_words()
        turn.ignore_disfluency()

        return turn

    def __annotation_generator(self, sentence: Turn, d_tag: List[DisfluencyEnum]):
        reparandum = ""
        correction = ""
        indices = []
        is_prev_disfluency = False

        for word, tag in zip(sentence.words, d_tag):
            if tag.name not in  ("O", "C"):
                is_prev_disfluency = True
                reparandum += word.text
                indices.append(word.idx)
                continue

            if is_prev_disfluency and tag.name == "C":
                correction += word.text

            elif is_prev_disfluency and tag.name == "O":
                disfluency_list = self.__classify_disfluency(reparandum, correction, indices)
                yield indices, disfluency_list

                reparandum = ""
                correction = ""
                indices = []
                is_prev_disfluency = False

        if is_prev_disfluency:
            disfluency_list = self.__classify_disfluency(reparandum, correction, indices)
            yield indices, disfluency_list

    def __classify_disfluency(self, reparandum: str, correction: str, indices: List[int]) -> List[int]:
        if reparandum == correction:
            disfluency_type = DisfluencyEnum.REPETITION
        else:
            disfluency_type = DisfluencyEnum.SELF_REPAIR

        return [disfluency_type for _ in indices]
