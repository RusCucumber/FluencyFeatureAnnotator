import json
from pathlib import Path
from time import sleep
from warnings import warn

import pandas as pd
from pydub import AudioSegment
from rev_ai import Job, Transcript, apiclient

FILLER = {
    "uh", "ah", "um", "mm",
    "hmm", "oh", "mm-hmm",
    "mmhmm", "er", "mhm",
    "uh-huh", "uhhuh", "er",
    "erm", "huh", "uhu"
}

def transcript_2_df(transcript: Transcript) -> pd.DataFrame:
    data = []
    for monologues in transcript.monologues:
        for elements in monologues.elements:
            row = {
                "start_time": elements.timestamp,
                "end_time": elements.end_timestamp,
                "text": elements.value,
                "type": elements.type_,
                "confidence": elements.confidence
            }

            data.append(row)

    return pd.DataFrame.from_dict(data)

def save_job(job: Job, save_path: Path) -> None:
    response = {
        "id": job.id,
        "created_on": job.created_on,
        "completed_on": job.completed_on,
        "metadeta": job.metadata,
        "failure_detail": job.failure_detail,
        "delete_after_seconds": job.delete_after_seconds,
        "status": job.status.name,
        "name": job.name,
        "duration_seconds": job.duration_seconds,
        "failure": job.failure,
        "media_url": job.media_url,
        "skip_diarization": job.skip_diarization,
        "skip_punctuation": job.skip_punctuation,
        "remove_disfluencies": job.remove_disfluencies,
        "filter_profanity": job.filter_profanity,
        "custom_vocabulary_id": job.custom_vocabulary_id,
        "speaker_channel_count": job.speaker_channels_count,
        "language": job.language,
        "transcriber": job.transcriber
    }

    with open(save_path, "w") as f:
        json.dump(response, f)

class RevAsr:
    def __init__(self, access_token: str, tmp_dir=None):
        self._client = apiclient.RevAiAPIClient(
            access_token
        )

        if tmp_dir is None:
            tmp_dir = Path(__file__).parent

        tmp_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = tmp_dir

    def __call__(self, audio_path: Path, wait=60, timeout=600, df=True):
        return self.recognize_speech(audio_path, wait, timeout, df)

    def check_audio(self, audio_path: Path) -> str:
        """
        Check if a target audio is longer than 2 sec because Rev TTS cannot recognize short (2sec >) speech
        """

        if not audio_path.exists():
            raise RuntimeError(f"audio_path {str(audio_path)} does not exist")

        audio = AudioSegment.from_file(audio_path)
        if audio.duration_seconds <= 2:
            # raise ValueError(f"audio {str(audio_path)} less than 2 sec")
            warn("Since audio duration less than 2 sec, padded silence.")

            audio_path = self.tmp_dir / "tmp.wav"
            audio += AudioSegment.silent(duration=2000)
            audio.export(audio_path, format="wav")

        return str(audio_path)

    def submit(self, audio_path: Path) -> Job:
        """
        Submit a job to Rev TTS server
        """

        if isinstance(audio_path, (str, Path)):
            audio_path = Path(audio_path)
        else:
            raise TypeError(f"audio_path must be str or Path, not {type(audio_path)}")

        audio_path = self.check_audio(audio_path)

        job = self._client.submit_job_local_file(
            audio_path,
            transcriber="machine_v2"
        )

        return job

    def recognize_speech(self, audio_path: Path, wait: int =60, timeout: int =600, df: bool =True) -> Transcript:
        """
        Submit a job and get a result
        - audio_path: Path ... file path of target speech data
        - wait: int ... wait time of each request (default 60)
        - timeout: int ... timeout of requests (default 600)
        - df: bool ... if df=True, return a transcription as a pandas.DataFrame object. if df=False, return a rev_ai.Transcript object
        """

        job = self.submit(audio_path)

        cur_wait_time = 0
        print("in progress", flush=True, end=" =")
        while True:
            sleep(wait)
            cur_wait_time += wait
            if cur_wait_time > timeout:
                print("\n")
                raise TimeoutError(f"job {job.id}: timeout")

            job_detail = self._client.get_job_details(job.id)
            status = job_detail.status.name

            if status == "IN_PROGRESS":
                print("=", flush=True, end="")
                continue

            elif status == "TRANSCRIBED":
                print("> finished!", flush=True)
                return self.load_transcription(job.id, df)

            else:
                print("\n")
                raise RuntimeError(f"job {job.id}: {job_detail.failure}")

    def is_job_completed(self, job_id: str) -> bool:
        job = self._client.get_job_details(job_id)
        status = job.status.name

        if status == "IN_PROGRESS":
            return False
        elif status == "TRANSCRIBED":
            return True
        else:
            raise RuntimeError(f"job {job.id} was failed: {job.failure}")

    def load_transcription(self, job_id: str, df=True) -> Transcript:
        transcript = self._client.get_transcript_object(job_id)

        if df:
            return transcript_2_df(transcript)
        else:
            return transcript

