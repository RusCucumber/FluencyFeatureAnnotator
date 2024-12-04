from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import parselmouth
from pydub import AudioSegment


class PraatVAD:
    def __init__(
        self,
        time_step: float =None,
        silence_threshold: float =-25.0,
        minimum_silence_interval_duration: float =0.25,
        minimum_sounding_interval_duration: float =0.25
    ) -> 'PraatVAD':
        self.time_step = time_step
        self.silence_threshold = silence_threshold
        self.minimum_silence_interval_duration = minimum_silence_interval_duration
        self.minimum_sounding_interval_duration = minimum_sounding_interval_duration

    def __call__(self, wav_path: Union[str, Path]) -> List[Tuple[float, float]]:
        return self.detect_voiced_timestamp(wav_path)

    def detect_voiced_timestamp(self, wav_path: Union[str, Path]) -> List[Tuple[float, float]]:
        silence_timestamp = self.detect_silence_timestamp(wav_path)
        duration = len(AudioSegment.from_wav(wav_path)) * 0.001

        silence_timestamp = [(0,0)] + silence_timestamp

        voiced_timestamp = []
        for t1, t2 in zip(silence_timestamp[:-1], silence_timestamp[1:]):
            timestamp = (t1[1], t2[0])
            voiced_timestamp.append(timestamp)

        if t2[1] != duration:
            voiced_timestamp.append(t2[1], duration)

        return voiced_timestamp

    def detect_silence_timestamp(self, wav_path: Union[str, Path]) -> List[Tuple[float, float]]:
        if isinstance(wav_path, Path):
            wav_path = str(wav_path)

        snd = parselmouth.Sound(wav_path)
        intensity = snd.to_intensity(time_step=self.time_step)
        t = intensity.xs()
        i = intensity.values[0]

        sound_indicies = np.where(i >= max(i) + self.silence_threshold)[0]
        interval_list = self.__find_appropriate_interval(t, sound_indicies, self.minimum_sounding_interval_duration)

        silence_timestamp = []

        for i, j in zip(interval_list[:-1], interval_list[1:]):
            dur = t[j[0]] - t[i[1]]
            if dur > self.minimum_silence_interval_duration:
                silence_timestamp.append((t[i[1]], t[j[0]]))

        return silence_timestamp

    def __find_appropriate_interval(
        self,
        t: List[float],
        ascending_order_indices: List[int],
        minimum_interval_duration: float
    ) -> List[Tuple[int, int]]:
        interval_list = []
        begin = ascending_order_indices[0]
        prev_idx = begin - 1

        for idx in ascending_order_indices:
            # 連続区間の終わりの場合
            if idx - 1 != prev_idx:
                duration =  t[prev_idx] - t[begin]
                # 連続区間の長さ[s]が規定以上の場合
                if duration >= minimum_interval_duration:
                    interval_list.append((begin, prev_idx))
                begin = idx
            prev_idx = idx

        if begin > 0:
            duration = t[idx] - t[begin]
            if duration >= minimum_interval_duration:
                interval_list.append((begin, idx))

        return interval_list

