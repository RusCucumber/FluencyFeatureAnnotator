from typing import List, Union
from warnings import warn

import pandas as pd
from google.cloud.speech import WordInfo
from modules.common.base import Base
from modules.common.clause import Clause
from modules.common.word import DisfluencyEnum, Word


class Turn(Base):
    def __init__(self, clause_list: List[Clause], idx: int) -> 'Turn':
        """
        - clause_list: list of Clause objects
        - idx: index of turn
        """
        super().__init__()
        self.text = clause_list
        self.idx = idx
        self.start_time = clause_list[0].words[0].start_time
        self.end_time = clause_list[-1].words[-1].end_time

        self.__clauses = clause_list
        self.__show_disfluency = True

        self.__words = self.__generate_words(clause_list, with_disfluency=True)
        self.__fluency_words = self.__generate_words(clause_list, with_disfluency=False)

    def __len__(self) -> int:
        return len(self.__clauses)

    @classmethod
    def from_google_asr_words(cls, words: List[WordInfo], idx: int) -> 'Turn':
        clause = Clause.from_google_asr_words(words, idx)

        return cls([clause], idx)

    @classmethod
    def from_DataFrame(cls, df: pd.DataFrame, idx: int, **kwargs) -> 'Turn':
        """
        Generate a Turn object
        - df: pandas DataFrame
        - idx: index of turn
        - word_col: column name of word texts
        - start_time_col: column name of start time information
        - end_time_col: column name of end time information
        """
        clause = Clause.from_DataFrame(df, idx, **kwargs)

        return cls([clause], idx)

    @Base.text.setter
    def text(self, clause_list: List[Clause]) -> None:
        text = ""
        for clause in clause_list:
            if type(clause) is Clause:
                text += (clause.text + " ")
            else:
                raise ValueError(f"Input must be the list of {Clause}")

        self._text = text[:-1]

    @property
    def words(self) -> List[Word]:
        if self.__show_disfluency:
            return self.__words

        return self.__fluency_words

    @property
    def clauses(self) -> List[Clause]:
        return self.__clauses

    def lower(self) -> None:
        for clause in self.clauses:
            clause.lower()

    def upper(self) -> None:
        for clause in self.clauses:
            clause.upper()

    def show_info(self) -> None:
        print(f"turn[{self._idx:03d}]: {super().__repr__()}")

    def __xor(self, a: bool, b: bool) -> bool:
        return (a and not b) or (b and not a)

    def __generate_words(self, clauses: List[Clause], with_disfluency: bool =True) -> List[Word]:
        is_state_reversed = False
        if self.__xor(self.__show_disfluency, with_disfluency):
            is_state_reversed = True

        words = []
        for clause in clauses:
            if with_disfluency:
                clause.show_disfluency()
            else:
                clause.ignore_disfluency()

            words += clause.words

            if is_state_reversed and self.__show_disfluency:
                clause.show_disfluency()
            else:
                clause.ignore_disfluency()

        return words

    def ignore_disfluency(self) -> None:
        if self.__show_disfluency:
            self.__show_disfluency = False
            for clause in self.__clauses:
                clause.ignore_disfluency()
            self.text = self.__clauses

    def show_disfluency(self) -> None:
        if not self.__show_disfluency:
            self.__show_disfluency = True
            for clause in self.__clauses:
                clause.show_disfluency()
            self.text = self.__clauses

    # TODO:
    ## Clause の言い淀み検知を行っても，Turn の words に反映されないため，仮に作成するメソッド
    ## スマートな方法に修正したい
    def reset_words(self) -> None:
        self.__words = self.__generate_words(self.__clauses, with_disfluency=True)
        self.__fluency_words = self.__generate_words(self.__clauses, with_disfluency=False)

    def separate_clause(self, idx_clause: int, indices_words: List[int], reset_idx: bool =False) -> None:
        if type(indices_words) is not list:
            ValueError("Input variable \"indices_words\" must be list of int")

        clause = self.__clauses[idx_clause]

        if len(clause) <= 1:
            return

        separated_clause_list = []
        for idx in indices_words:
            try:
                separeated_clause = clause.split_words(idx)
                if not self.__show_disfluency:
                    separeated_clause.ignore_disfluency()

            except KeyError:
                warn(f"specified idx {idx} is end word of clause")
                continue
            except Exception as e:
                raise e

            if reset_idx:
                clause.reset_index()

            separated_clause_list.append(clause)
            clause = separeated_clause

        if reset_idx:
            clause.reset_index()
        separated_clause_list.append(clause)

        front = self.__clauses[:idx_clause]
        behind = self.__clauses[idx_clause + 1:]

        idx_behind = separated_clause_list[-1].idx + 1
        for i, clause in enumerate(behind):
            clause.idx = idx_behind + i

        self.__clauses = front + separated_clause_list + behind

    def search_clause_idx(self, word_idx: int, start_from: int =0) -> Union[int, None]:
        for clause in self.__clauses[start_from:]:
            for word in clause.words:
                if word.idx == word_idx:
                    return clause.idx

        for clause in self.__clauses[:start_from]:
            for word in clause.words:
                if word.idx == word_idx:
                    return clause.idx

        return None

def shorten_turn(turn: Turn, end_time: float) -> Turn:
    turn.show_disfluency()

    clause_short_list = []
    find_end_time = False
    for clause in turn.clauses:
        clause_short = []

        for word in clause.words:

            # ある単語が，指定された end_time よりもあとの時刻に開始している場合，ループを終了
            if word.start_time >= end_time:
                find_end_time = True
                break

            clause_short.append(word)

        if len(clause_short) > 0:
            clause_short = Clause(clause_short, clause.idx)
            clause_short_list.append(clause_short)

        if find_end_time:
            break

    turn_short = Turn(clause_short_list, 0)

    return turn_short

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

    turn = Turn.from_google_asr_words(words, 0)

    for word in turn.words:
        word.show_info()

    turn.separate_clause(0, [9])

    turn.separate_clause(0, [4])

    turn.clauses[0].annotate_disfluency([3], [DisfluencyEnum.SELF_REPAIR])
    turn.clauses[1].annotate_disfluency([5], [DisfluencyEnum.SELF_REPAIR])

    turn.ignore_disfluency()

    turn.show_info()

    for clause in turn.clauses:
        clause.show_info()
        clause.show_disfluency()

    turn.show_info()

    turn.show_disfluency()

    turn.show_info()
