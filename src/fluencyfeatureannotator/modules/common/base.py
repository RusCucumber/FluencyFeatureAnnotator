import warnings


# Wordクラスの基底クラス
class Base:
    def __init__(self) -> 'Base':
        self._text = ""
        self._idx = -1
        self._start_time = -1
        self._end_time = -1

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return f"{self._text} ({self._start_time:.3f}:{self._end_time:.3f})"

    def __len__(self) -> int:
        return len(self._text)

    @property
    def idx(self) -> int:
        return self._idx

    @idx.setter
    def idx(self, idx: int) -> None:
        if type(idx) is int:
            if idx < 0:
                raise(IndexError("idx must be bigger than -1"))
            self._idx = idx

        else:
            raise(TypeError("index must be int"))

    @idx.deleter
    def idx(self) -> None:
        self._idx = -1

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, text: str) -> None:
        if type(text) is str:
            if text == "":
                warnings.warn("Inputed String is empty (\"\"). Ignore this input.", category=UserWarning)
            else:
                self._text = text

        else:
            raise(TypeError("index must be String"))

    @property
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, start_time: float) -> None:
        if type(start_time) in (float, int):
            if start_time < 0:
                raise ValueError("start_time must be bigger than -1")

            if self._end_time != -1 and start_time > self._end_time:
                raise ValueError("start_time must be bigger than end_time")

            self._start_time = float(start_time)
        else:
            raise TypeError("start_time must be float or int")

    @property
    def end_time(self) -> None:
        return self._end_time

    @end_time.setter
    def end_time(self, end_time: float) -> None:
        if type(end_time) in (float, int):
            if end_time < 0:
                raise ValueError("end_time must be bigger than -1")

            if end_time < self._start_time:
                raise ValueError("end_time must be bigger than start_time")

            self._end_time = float(end_time)
        else:
            raise TypeError("end_time must be float or int")

    def get_duration(self) -> None:
        return self._end_time - self._start_time

    def lower(self) -> None:
        self._text = self._text.lower()

    def upper(self) -> None:
        self._text = self._text.upper()

    def show_info(self) -> None:
        print(repr(self))

