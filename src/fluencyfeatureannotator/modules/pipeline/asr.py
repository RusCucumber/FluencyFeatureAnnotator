from pathlib import Path
from typing import Any, Union

from modules.common import Turn
from modules.models import GoogleTranscriptGenerator
from modules.pipeline.utils.base import ModuleBase
from modules.pipeline.utils.exceptions import ASRError


class ASR(ModuleBase):
    def __init__(self, frame_rate: int =16000) -> 'ASR':
        super().__init__()
        self._model = GoogleTranscriptGenerator(frame_rate=frame_rate)

    @ModuleBase.model.setter
    def model(self, config):
        self._model.config = config

    def __call__(self, wav: Union[str, Path]) -> Turn:
        return self.predict(wav)

    def predict(self, wav: Any, format: str ="wav", **kwargs) -> Turn:
        if format == "wav":
            results = self._model.from_wav(wav)
        elif format == "gcs":
            results = self._model.from_gcs(wav, **kwargs)
        elif format == "audiosegment":
            results = self._model.from_audiosegment(wav)
        else:
            raise ASRError(f"ASR does not supported {format} format audio")

        if len(results["words"]) == 0:
            raise ASRError(f"ASR result is {results}. Speech is not recognized.")

        turn = Turn.from_google_asr_words(results["words"], 0)

        return turn

