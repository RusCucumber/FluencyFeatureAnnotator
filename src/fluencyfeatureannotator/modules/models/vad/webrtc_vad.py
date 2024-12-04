from collections import deque
from pathlib import Path
from typing import Generator, List, Tuple, Union

import webrtcvad
from pydub import AudioSegment


class WebRTCVAD:
    def __init__(
        self,
        hangover_time: int =300,
        aggressiveness: int =1,
        t_frame: int =30,
        frame_rate: int =16000
    ) -> 'WebRTCVAD':
        """
        Voice Activity Detector with hangover time
        """
        if t_frame not in (10, 20, 30):
            raise ValueError("t_frame must be 10, 20 or 30")

        if aggressiveness not in (1, 2, 3):
            raise ValueError("aggressiveness must be 1, 2 or 3")

        if frame_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("frame_rate must be 8000, 16000, 32000, 48000")

        self._vad = webrtcvad.Vad(aggressiveness)
        self._hangover_time = hangover_time
        self._t_frame = t_frame # ms: webrtcvad only accepts frame of 10ms, 20ms and 30ms
        self._frame_rate = frame_rate
        self.__sample_width = 2
        self.__chunk_size = int(self._frame_rate / 1000 * self._t_frame)
        self.__stride = self.__chunk_size * self.__sample_width

        window_len = int(self._hangover_time / self._t_frame)
        self.__vad_window = deque(maxlen=window_len)

        self.__audio_buffer = b''
        self._is_speech_section = False

    @property
    def vad(self) -> webrtcvad.Vad:
        return self._vad

    @vad.setter
    def vad(self, aggressiveness: int) -> None:
        if aggressiveness not in (1, 2, 3):
            raise ValueError("aggressiveness must be 1, 2 or 3")
        self._vad = webrtcvad.Vad(aggressiveness)

    @property
    def hangover_time(self):
        return self._hangover_time

    @hangover_time.setter
    def hangover_time(self, hangover_time: int) -> None:
        self._hangover_time = hangover_time

        window_len = int(self._hangover_time / self._t_frame)
        self.__vad_window = deque(maxlen=window_len)

    @property
    def t_frame(self) -> int:
        return self._t_frame

    @t_frame.setter
    def t_frame(self, t_frame: int) -> None:
        if t_frame not in (10, 20, 30):
            raise ValueError("t_frame must be 10, 20 or 30")
        self._t_frame = t_frame

        self.__chunk_size = int(self._frame_rate / 1000 * self._t_frame)
        self.__stride = self.__chunk_size * self.__sample_width

        window_len = int(self.hangover_time / self._t_frame)
        self.__vad_window = deque(maxlen=window_len)

    @property
    def frame_rate(self) -> int:
        return self._frame_rate

    @frame_rate.setter
    def frame_rate(self, frame_rate: int) -> None:
        if frame_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("frame_rate must be 8000, 16000, 32000, 48000")
        self._frame_rate = frame_rate

        self.__chunk_size = int(self._frame_rate / 1000 * self._t_frame)
        self.__stride = self.__chunk_size * self.__sample_width

    def __call__(self, wav_path: Union[str, Path]) -> List[Tuple[float, float]]:
        return self.detect_voiced_timestamp(wav_path)

    def is_speech_section(self, audio: bytes) -> bool:
        """
        detects speech section
        """

        # generate chunk readable for webrtc vad
        self.__audio_buffer += audio
        while len(self.__audio_buffer) > 0:
            audio_chunk = self.__audio_buffer[:self.__stride]
            self.__audio_buffer = self.__audio_buffer[self.__stride:]
            if len(audio_chunk) != self.__chunk_size * self.__sample_width:
                break
            is_speech = self._vad.is_speech(audio_chunk, self._frame_rate)
            self.__vad_window.append(is_speech)

        # is speech sectioin if above 90% of frame in window is predicted voiced
        if not self._is_speech_section and sum(self.__vad_window) >= 0.9 * self.__vad_window.maxlen:
            self._is_speech_section = True
        elif self._is_speech_section and sum(self.__vad_window) <= 0.1 * self.__vad_window.maxlen:
            self._is_speech_section = False

        return self._is_speech_section

    def detect_voiced_timestamp(self, wav_path: Union[str, Path]) -> List[Tuple[float, float]]:
        audio = AudioSegment.from_wav(wav_path)

        if audio.sample_width != 2:
            raise ValueError("sample width of audio data must be 16 bit (= 2 byte)")

        if audio.frame_rate != self._frame_rate:
            audio = audio.set_frame_rate(self._frame_rate)

        audio_buffer = b''
        was_speech_section = False
        voiced_timestamp = []
        for audio_chunk, ts in self.__chunk_generator(audio.raw_data):
            is_speech_section = self.is_speech_section(audio_chunk)
            if is_speech_section:
                audio_buffer += audio_chunk
                if not was_speech_section:
                    start_time = ts
            elif was_speech_section:
                voiced_timestamp.append((start_time, ts))
                audio_buffer = b''

            was_speech_section = is_speech_section

        if is_speech_section and len(audio_buffer)>0:
            voiced_timestamp.append((start_time, ts))

        return voiced_timestamp

    def detect_silence_timestamp(self, wav_path: Union[str, Path]) -> List[Tuple[float, float]]:
        voiced_timestamp = self.detect_voiced_timestamp(wav_path)

        silence_timestamp = []
        for ts1, ts2 in zip(voiced_timestamp[:-1], voiced_timestamp[1:]):
            silence_timestamp.append((ts1[1], ts2[0]))

        return silence_timestamp

    def __chunk_generator(self, audio: bytes) -> Generator[Tuple[bytes, float], None, None]:
        n_chunk = 0
        while len(audio) > 0:
            chunk = audio[:self.__stride]
            ts = self._t_frame * n_chunk * 0.001
            yield chunk, ts
            audio = audio[self.__stride:]
            n_chunk += 1

