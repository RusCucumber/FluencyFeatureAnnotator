from modules.common import Turn
from modules.models import PunctuationPredictor
from modules.pipeline.utils.base import ModuleBase


class SentenceEndDetector(ModuleBase):
    def __init__(self, device: str ="cpu") -> 'SentenceEndDetector':
        super().__init__()
        self._model = PunctuationPredictor.from_finetuned(device=device)

    def predict(self, turn: Turn) -> Turn:
        self._turn_type_checker(turn)

        # TODO: add turn が全て disfluency の場合，以下の処理をしないようにする

        text = []
        is_all_word_disfluent = True # turn を構成する全ての単語が言い淀みの場合，True になる
        for word in turn.words:
            text.append(word.text)

            if word.idx != -1:
                is_all_word_disfluent = False

        if is_all_word_disfluent:
            return turn

        p_tags = self._model(text)

        indices = []
        for idx, p_tag in enumerate(p_tags[0]):
            if p_tag.name == "I":
                indices.append(idx)

        if idx in indices:
            indices.remove(idx)

        turn.separate_clause(0, indices)

        return turn
