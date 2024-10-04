from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch

torch.set_num_threads(1)

class SileroVAD:
    def __init__(
        self,
        frame_rate: int =16000,
        min_speech_duration: float =0.25,
        min_silence_duration: float =0.25,
        t_frame: float =0.1,
        t_step: float =0.02,
        model_name: str ="silero_vad",
        forced_reload: bool =False
    ) -> 'SileroVAD':
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model=model_name,
            force_reload=forced_reload
        )
        (get_speech_ts, _, _, read_audio, _, _, _) = utils

        self._vad = model
        self.__get_speech_ts = get_speech_ts
        self.__read_audio = read_audio

        self.frame_rate = frame_rate
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.t_frame = t_frame
        self.t_step = t_step

    @property
    def vad(self) -> torch.Module:
        return self._vad

    @vad.setter
    def vad(self, model_name: str) -> None:
        if not isinstance(model_name, str):
            ValueError("model must be str")

        model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model=model_name, force_reload=False)
        (get_speech_ts, _, _, read_audio, _, _, _) = utils

        self._vad = model
        self.__get_speech_ts = get_speech_ts
        self.__read_audio = read_audio


    def __call__(self, wav_path: Union[str, Path]) -> List[Tuple[float, float]]:
        return self.detect_voiced_timestamp(wav_path)

    def __calc_parameters(self) -> Dict[str, int]:
        parameters = {}

        parameters["num_steps"] = int(self.t_frame / self.t_step)
        parameters["min_speech_samples"] = int(self.frame_rate * self.min_speech_duration)
        parameters["min_silence_samples"] = int(self.frame_rate * self.min_silence_duration)
        parameters["num_samples_per_window"] = int(self.frame_rate * self.t_frame)

        return parameters

    def detect_voiced_timestamp(self, wav_path: Union[str, Path]) -> List[Tuple[float, float]]:
        kwargs = self.__calc_parameters()

        wav = self.__read_audio(wav_path, target_sr=self.frame_rate)
        timestamps = self.__get_speech_ts(wav, self._vad, **kwargs)

        voiced_timestamps = []
        for ts in timestamps:
            start_time = ts["start"] / self.frame_rate
            end_time = ts["end"] / self.frame_rate

            voiced_timestamps.append((start_time, end_time))

        return voiced_timestamps

    def detect_silence_timestamp(self, wav_path: Union[str, Path]) -> List[Tuple[float, float]]:
        voiced_timestamp = self.detect_voiced_timestamp(wav_path)

        silence_timestamp = []
        for ts1, ts2 in zip(voiced_timestamp[:-1], voiced_timestamp[1:]):
            silence_timestamp.append((ts1[1], ts2[0]))

        return silence_timestamp

