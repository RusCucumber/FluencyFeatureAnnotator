from pathlib import Path
from typing import Union

import pydub


class GoogleTranscriptGenerator:
    from google.cloud import speech

    def __init__(self,
        google_api_key_json_path: Union[str, Path],
        frame_rate: int =16000
    ) -> 'GoogleTranscriptGenerator':
        self.asr_client = self.speech.SpeechClient.from_service_account_json(google_api_key_json_path)

        self._config = {
            "encoding" : self.speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "sample_rate_hertz" : frame_rate,
            "language_code" : "en-US",
            "enable_word_time_offsets" : True
        }

        self.asr_config = self.speech.RecognitionConfig(**self._config)

        self.__frame_rate = frame_rate

    @property
    def config(self) -> None:
        return self._config

    @config.setter
    def config(self, config: dict) -> None:
        if not isinstance(config, dict):
            raise ValueError("config must be dict")

        for key, value in config.items():
            self._config[key] = value

        self.asr_client = self.speech.RecognitionConfig(**self._config)


    def from_audiosegment(self, audiosegment: pydub.AudioSegment) -> dict:
        if audiosegment.frame_rate != self.__frame_rate:
            audiosegment = audiosegment.set_frame_rate(self.__frame_rate)

        stream = [audiosegment.raw_data]

        requests = (
            self.speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream
        )

        streaming_config = self.speech.StreamingRecognitionConfig(config=self.asr_config)

        # streaming_recognize returns a generator.
        responses = self.asr_client.streaming_recognize(
            config=streaming_config,
            requests=requests,
        )

        results = {
            "transcript": [],
            "words": []
        }
        for response in responses:
            # Once the transcription has settled, the first result will contain the
            # is_final result. The other results will be for subsequent portions of
            # the audio.
            for result in response.results:
                alternative = result.alternatives[0]
                results["transcript"].append(alternative.transcript)
                results["words"] += alternative.words

        results["transcript"] = " ".join(results["transcript"])
        return results


    def from_wav(self, wav_path: Union[str, Path]) -> dict:
        audiosegment = pydub.AudioSegment.from_wav(wav_path)

        return self.from_audiosegment(audiosegment)


    def from_gcs(self, gcs_uri: str, timeout: int =90) -> dict:
        audio = self.speech.RecognitionAudio(uri=gcs_uri)

        operation = self.asr_client.long_running_recognize(
            config=self.asr_config,
            audio=audio
        )

        response = operation.result(timeout=timeout)

        results = {
            "transcript": [],
            "words": []
        }
        for result in response.results:
            alternative = result.alternatives[0]
            results["transcript"].append(alternative.transcript)
            results["words"] += alternative.words

        results["transcript"] = " ".join(results["transcript"])
        return results

