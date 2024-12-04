from pathlib import Path
from typing import List, Tuple, Union

from modules.common import Turn
from modules.models import PraatVAD, SileroVAD, WebRTCVAD
from modules.pipeline.utils.base import ModuleBase
from modules.pipeline.utils.pause_location import PauseLocation


class PauseLocationDetector(ModuleBase):
    def __init__(self, model: str ="webrtc", vad_config: dict ={}) -> 'PauseLocationDetector':
        super().__init__()
        self._model = self.__select_vad(model, vad_config)

    def __call__(self, wav_path: Union[str, Path], turn: Turn) -> List[dict]:
        return self.predict(wav_path, turn)

    def predict(self, wav_path, turn:Turn) -> List[dict]:
        silence_ts = self._model.detect_silence_timestamp(wav_path)

        # no silence
        if len(silence_ts) == 0:
            return []

        clause_ts = self.estimate_clause_ts(turn)

        pause_location = self.estimate_pause_location(silence_ts, clause_ts)

        result = []
        for pl, (start, end) in zip(pause_location, silence_ts):
            pause = {
                "location": pl,
                "start_time": start,
                "end_time": end
            }
            result.append(pause)

        return result

    def estimate_clause_ts(self, turn:Turn) -> List[Tuple[float, float]]:
        self._turn_type_checker(turn)

        clause_ts = []
        for clause in turn.clauses:
            start_time = clause.start_time
            end_time = clause.end_time
            clause_ts.append((start_time, end_time))

        return clause_ts

    def estimate_pause_location(
        self,
        silence_ts: List[Tuple[float, float]],
        clause_ts: List[Tuple[float, float]]
    ) -> List[PauseLocation]:
        pause_location = [0 for _ in silence_ts]

        # TODO: CI, CE を判定する処理の作成
        pointer = 0
        for c_start, c_end in clause_ts:
            for i, (p_start, p_end) in enumerate(silence_ts[pointer:]):
                # ポーズが節内に出現　→　CI
                if p_start > c_start and p_end < c_end:
                    pause_location[i + pointer] = 1
                # ポーズが現在の節より外で出現　→　ループを一時終了し，次の節で再開
                if p_start >= c_end:
                    pointer = i
                    break

        pause_location = self.__transform_2_enum(pause_location)

        return pause_location

    def __transform_2_enum(self, pause_location_int: List[int]) -> List[PauseLocation]:
        pause_location_enum = []
        for pl in pause_location_int:
            if pl == 0:
                pause_location_enum.append(PauseLocation.CLAUSE_EXTERNAL)
            elif pl == 1:
                pause_location_enum.append(PauseLocation.CLAUSE_INTERNAL)

        return pause_location_enum

    def __select_vad(self, model: str, vad_config: dict) -> Union[PraatVAD, WebRTCVAD, SileroVAD]:
        assert isinstance(vad_config, dict), "configuration of VAD must be dict"

        if model == "praat":
            return PraatVAD(**vad_config)
        if model == "webrtc":
            return WebRTCVAD(**vad_config)
        if model == "silero":
            return SileroVAD(**vad_config)

        raise ValueError("Currently \"praat\", \"webrtc\" and \"silero\" are supported VAD")

