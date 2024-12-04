from pathlib import Path
from typing import Generator, Tuple, Union

import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

"""
以下のようなディレクトリ構成を前提とする
なお，tsv ファイルは既に turn 単位の書き起こしへ変換されているものとする
- transcript
    |- 001.tsv
    |- 002.tsv
    :
- movie
    |- 001.mp4
    |- 002.mp4
    :
"""

# 自分の環境に合わせてパスを変更（絶対パス推奨)
TRANSCRIPT_DIR = Path("transcript")
INTERVIEW_DIR = Path("movie")
SAVE_DIR = Path("output")

# tsv ファイルがあるディレクトリから tsv パスを自動で読み込む関数
def transcript_path_generator(transcript_dir: Union[Path, str]) -> Generator[Path, None, None]:
    """
    - transcript_dir: Path|str ... 書き起こし (tsvファイル) があるディレクトリ
    """
    if isinstance(transcript_dir, str):
        transcript_dir = Path(transcript_dir)

    for transcript_tsv_path in transcript_dir.glob("*.tsv"):
        yield transcript_tsv_path

# mp4 ファイルがあるディレクトリから transcript_dir と同じユーザid の mp4 ファイルを読み込む関数
def get_interview_path(interview_dir: Union[Path, str], transcript_tsv_path: Path) -> Path:
    """
    - interview_dir: Path|str ... インタビュー動画 (mp4ファイル) があるディレクトリ
    """
    if isinstance(interview_dir, str):
        interview_dir = Path(interview_dir)

    interview_mp4_path = interview_dir / f"{transcript_tsv_path.stem}.mp4"
    if interview_mp4_path.exists():
        return interview_mp4_path

    raise FileNotFoundError(f"{interview_mp4_path} does not found.")

# tsv & mp4 パスから，データを読み込む関数
def load_interview_audio_transcript(
    transcript_tsv_path: Path,
    interview_mp4_path: Path
) -> Tuple[pd.DataFrame, AudioSegment]:
    df_transcript = pd.read_table(transcript_tsv_path)
    audio_interview = AudioSegment.from_file(interview_mp4_path)

    return df_transcript, audio_interview

# hh:mm:ss.ms → ms duration へ変換する関数
def timestamp_2_msec_duration(timestamp: str) -> int:
    h, m, s = timestamp.split(":")
    s, ms = s.split(".")

    duration = (h * 3600 + m * 60 + s + ms) * 1000

    return int(duration)

# 書き起こしの時間情報を基に，音声ファイルを分割する関数
def split_audio_generator(
    df_transcript: pd.DataFrame,
    audio_interview: AudioSegment
) -> Generator[AudioSegment, None, None]:
    for idx in tqdm(df_transcript.index, desc="spliting interview movie in turn_wise audio"):
        if df_transcript.at[idx, "speaker"] == "system":
            continue

        start_timestamp = df_transcript.at[idx, "start_time"]
        end_timestamp = df_transcript.at[idx, "end_time"]

        start_duration = timestamp_2_msec_duration(start_timestamp)
        end_duration = timestamp_2_msec_duration(end_timestamp)

        turn_wise_audio = audio_interview[start_duration:end_duration]

        yield turn_wise_audio

# 分割した音声ファイルを保存する関数
def save_audio(audio: AudioSegment, save_path: Path) -> None:
    audio.export(save_path, format="wav")

def main() -> None:
    for transcript_tsv_path in transcript_path_generator(TRANSCRIPT_DIR):
        interview_mp4_path = get_interview_path(INTERVIEW_DIR, transcript_tsv_path)

        df_transcript, audio_interview = load_interview_audio_transcript(transcript_tsv_path, interview_mp4_path)

        for idx, turn_wise_audio in enumerate(split_audio_generator(df_transcript, audio_interview)):
            save_path = SAVE_DIR / f"{audio_interview.stem}_{idx}.wav"
            save_audio(turn_wise_audio, save_path)

if __name__ == "__main__":
    main()

