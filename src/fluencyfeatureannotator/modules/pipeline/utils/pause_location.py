from enum import Enum, auto


class PauseLocation(Enum):
    CLAUSE_INTERNAL = auto()
    CLAUSE_EXTERNAL = auto()
    BEFORE_TURN = auto()
    AFTER_TURN = auto()
    CLAUSE_COVERED = auto()

