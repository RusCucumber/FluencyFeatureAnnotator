from typing import Any

from modules.common import Turn


class ModuleBase:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        return repr(self._model)

    def __call__(self, turn: Turn) -> Any:
        return self.predict(turn)

    def predict(self, turn: Turn) -> Any:
        return None

    def _turn_type_checker(self, turn: Turn) -> Any:
        if not isinstance(turn, Turn):
            raise ValueError(f"turn must be {Turn}")
