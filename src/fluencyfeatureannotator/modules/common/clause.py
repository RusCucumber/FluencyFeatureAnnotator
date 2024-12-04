from typing import List, Union

import pandas as pd
from google.cloud.speech import WordInfo
from modules.common.base import Base
from modules.common.word import DisfluencyEnum, DisfluencyWord, Word


class Clause(Base):
    def __init__(self, word_list: List[Word], idx: int) -> 'Clause':
        """
        - word_list: list of Word objectss
        - idx: index of clause
        """
        super().__init__()
        self.text = word_list
        self.idx = idx
        self.start_time = word_list[0].start_time
        self.end_time = word_list[-1].end_time

        self.__words = word_list
        self.__fluency_words = self.__generate_fluency_words()

        self.__show_disfluency = True

    def __len__(self) -> int:
        return len(self.__words)

    @classmethod
    def from_google_asr_words(cls, words: List[WordInfo], idx: int) -> 'Clause':
        """
        Generate a Clause object from a Google STT result
        - words: list of WordInfo objects
        - idx: index of clause
        """
        word_list = []
        for i, word_info in enumerate(words):
            word = Word.from_WordInfo(word_info, i)
            word_list.append(word)

        return cls(word_list, idx)

    @classmethod
    def from_DataFrame(
        cls,
        df: pd.DataFrame,
        idx: int,
        word_col: str ="word",
        start_time_col: str ="start_time",
        end_time_col: str ="end_time"
    ) -> 'Clause':
        """
        Generate a Clause object from Pandas DataFrame
        - df: DataFrame object
        - idx: index of clause
        - word_col: column name of word texts
        - start_time_col: column name of start time information
        - end_time_col: column name of end time information
        """
        word_list = []
        for i, row in df.iterrows():
            word = Word(row[word_col], i, row[start_time_col], row[end_time_col])
            word_list.append(word)

        return cls(word_list, idx)

    @Base.text.setter
    def text(self, word_list: List[Union[Word, DisfluencyWord]]) -> None:
        text = ""
        for word in word_list:
            if type(word) in (Word, DisfluencyWord):
                text += (word.text + " ")
            else:
                raise ValueError(f"Input must be the list of {Word}")

        self._text = text[:-1]

    @property
    def words(self) -> List[Word]:
        if self.__show_disfluency:
            return self.__words

        return self.__fluency_words

    def lower(self) -> None:
        for word in self.words:
            word.lower()

    def upper(self) -> None:
        for word in self.words:
            word.upper()

    def show_info(self) -> None:
        print(f"clause[{self._idx:03d}]: {super().__repr__()}")

    def __generate_fluency_words(self) -> List[Word]:
        return [word for word in self.__words if isinstance(word, Word)]

    def ignore_disfluency(self) -> None:
        if self.__show_disfluency:
            self.__show_disfluency = False
            self.text = self.__fluency_words

    def show_disfluency(self) -> None:
        if not self.__show_disfluency:
            self.__show_disfluency = True
            self.text = self.__words

    def annotate_disfluency(self, indices: List[int], disfluency_list: List[DisfluencyEnum]) -> None:
        if type(indices) is not list:
            raise ValueError("Input variable \"indices\" must be int or list of int")

        if type(disfluency_list) is not list:
            raise ValueError(f"Input variable \"disfluency_list\" must be int or list of {DisfluencyEnum}")

        idx_adjuster = 0
        for i, word in enumerate(self.__words):
            if word.idx in indices:
                disfluency_word = word.to_disfluency(disfluency_list[idx_adjuster])
                self.__words[i] = disfluency_word

                indices.remove(word.idx)
                idx_adjuster += 1
                continue

            if isinstance(word, Word):
                    word.idx -= idx_adjuster

        self.__fluency_words = self.__generate_fluency_words()

        if not self.__show_disfluency:
            self.text = self.__fluency_words

    def split_words(self, idx: int) -> 'Clause':
        if idx == self.__words[-1].idx:
            raise KeyError("specified idx is last word of clause. It is impossible separate")

        i_cb = 0
        for i, word in enumerate(self.__words):
            if word.idx == idx:
                i_cb = i + 1
                break
        i_cb_f = idx - self.__fluency_words[0].idx + 1

        cb_behind_words = self.__words[i_cb:]

        del self.__words[i_cb:]
        del self.__fluency_words[i_cb_f:]

        self.end_time = self.__words[-1].end_time

        if self.__show_disfluency:
            self.text = self.__words
        else:
            self.text = self.__fluency_words

        return Clause(cb_behind_words, self._idx + 1)

    def reset_index(self, idx: int =0) -> None:
        for word in self.__words:
            if isinstance(word, Word):
                word.idx = idx
                idx += 1

        self.__fluency_words = self.__generate_fluency_words()


if __name__ == "__main__":
    from datetime import timedelta

    words = [
        WordInfo(word="I", start_time=timedelta(seconds=0), end_time=timedelta(seconds=1)),
        WordInfo(word="am", start_time=timedelta(seconds=1), end_time=timedelta(seconds=2)),
        WordInfo(word="a", start_time=timedelta(seconds=2), end_time=timedelta(seconds=3)),
        WordInfo(word="st-", start_time=timedelta(seconds=3), end_time=timedelta(seconds=4)),
        WordInfo(word="student", start_time=timedelta(seconds=4), end_time=timedelta(seconds=6)),
        WordInfo(word="He", start_time=timedelta(seconds=6), end_time=timedelta(seconds=7)),
        WordInfo(word="she", start_time=timedelta(seconds=7), end_time=timedelta(seconds=8)),
        WordInfo(word="is", start_time=timedelta(seconds=8), end_time=timedelta(seconds=9)),
        WordInfo(word="my", start_time=timedelta(seconds=9), end_time=timedelta(seconds=10)),
        WordInfo(word="friend", start_time=timedelta(seconds=10), end_time=timedelta(seconds=11)),
        WordInfo(word="and", start_time=timedelta(seconds=11), end_time=timedelta(seconds=12)),
        WordInfo(word="she", start_time=timedelta(seconds=12), end_time=timedelta(seconds=13)),
        WordInfo(word="is", start_time=timedelta(seconds=13), end_time=timedelta(seconds=14)),
        WordInfo(word="so", start_time=timedelta(seconds=14), end_time=timedelta(seconds=15)),
        WordInfo(word="cute", start_time=timedelta(seconds=15), end_time=timedelta(seconds=16)),
    ]

    clause = Clause.from_google_asr_words(words, 0)

    for word in clause.words:
        print(word)

    clause.show_info()

    clause.annotate_disfluency([3, 5], [DisfluencyEnum.SELF_REPAIR, DisfluencyEnum.SELF_REPAIR])

    clause.show_info()

    clause.ignore_disfluency()

    for word in clause.words:
        print(word)

    clause.show_info()

    clause_2 = clause.split_words(3)

    clause.show_info()
    clause_2.show_info()

    #clause_2.reset_index()
    for word in clause_2.words:
        print(word.idx)

    clause_2.ignore_disfluency()
    clause_2.show_info()

    clause_3 = clause_2.split_words(7)
    clause_2.show_info()
    clause_3.show_info()

