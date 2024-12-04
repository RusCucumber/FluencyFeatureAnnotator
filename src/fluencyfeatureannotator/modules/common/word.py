from datetime import timedelta
from enum import Enum
from warnings import warn

from google.cloud.speech import WordInfo
from modules.common.base import Base


class DisfluencyEnum(Enum):
    REPETITION = 1
    SELF_REPAIR = 2
    FILLER = 3
    FALSE_START = 4
    OTHER = 5

# Wordクラスの基底クラス
class WordBase(Base):
    def __init__(self, text: str, idx: int, start_time: float, end_time: float) -> 'WordBase':
        """
        - text: word text
        - idx: word index
        - start_time: start time of word (sec)
        - end_time: end time of word (sec)
        """
        super().__init__()
        self.text = text
        self.idx = idx
        self.start_time = start_time
        self.end_time = end_time

    @Base.text.setter
    def text(self, text: str) -> None:
        if text == "":
            warn("Inputed String is empty (\"\"). Ignore this input", category=UserWarning)
            return

        if " " in text:
            warn("Inputed String looks more than two words", category=UserWarning)

        if type(text) is str:
            self._text = text
        else:
            raise(TypeError("index must be String"))

# DisfluencyWord クラス
class DisfluencyWord(WordBase):
    def __init__(self, text: str, start_time: float, end_time: float) -> 'DisfluencyWord':
        super().__init__(text, -1, start_time, end_time)
        del(self.idx)
        self._disfluency = DisfluencyEnum.OTHER

    @WordBase.idx.setter
    def idx(self, idx):
        if idx != -1:
            warn("idx of DisfluencyWord is always -1", category=UserWarning)

    @property
    def disfluency(self):
        return self._disfluency

    @disfluency.setter
    def disfluency(self, disfluency):
        if isinstance(disfluency, DisfluencyEnum):
            self._disfluency = disfluency
        else:
            raise(TypeError("disfluency must be DisfluencyEnum"))

    def show_info(self):
        print(f"word[{self._disfluency.name}]: {super().__repr__()}")

# Wordクラス
class Word(WordBase):
    def __init__(self, text: str, idx: int, start_time: float, end_time: float) -> 'Word':
        """
        - text: word text
        - idx: word index
        - start_time: start time of word (sec)
        - end_time: end time of word (sec)
        """
        super().__init__(text, idx, start_time, end_time)

    @classmethod
    def from_WordInfo(cls, word: WordInfo, idx: int):
        if isinstance(word, WordInfo):
            start_time = cls.__timedelta_2_sec(cls, word.start_time)
            end_time = cls.__timedelta_2_sec(cls, word.end_time)
            w = word.word
            if w[-1] == ".":
                w = w[:-1]
            return cls(w, idx, start_time, end_time)

        raise TypeError(f"input variable must be {WordInfo}")

    def to_disfluency(self, disfluency: DisfluencyEnum) -> DisfluencyWord:
        disfluency_word = DisfluencyWord(self.text, self.start_time, self.end_time)
        disfluency_word.disfluency = disfluency

        return disfluency_word

    def __timedelta_2_sec(self, timedelta: timedelta) -> float:
        sec = timedelta.seconds
        micsec = timedelta.microseconds
        return sec + (micsec * 1e-6)

    def show_info(self) -> None:
        print(f"word[{self._idx:03d}]: {super().__repr__()}")

if __name__ == "__main__":
    from datetime import timedelta

    s = timedelta(seconds=1)
    e = timedelta(seconds=2)

    word_info = WordInfo(word="hello", start_time=s, end_time=e)

    word = Word.from_WordInfo(word_info, 0)
    word.show_info()

    word = word.to_disfluency(DisfluencyEnum.REPETITION)
    #word.idx = 3
    word.show_info()

