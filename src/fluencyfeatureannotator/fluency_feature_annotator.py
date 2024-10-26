from pathlib import Path
from typing import Generator, List, Tuple, Union

import pandas as pd
from modules import Annotator, DisfluencyEnum, Turn, UtteranceFluencyMeasureExtractor
from modules.forced_alignment.utils.rev.df_2_rev import df_2_rev
from modules.forced_alignment.wav2vec_fa import Wav2VecCTCFA
from modules.pipeline.utils.pause_location import PauseLocation
from modules.utils.rev_utils import FILLER, transcript_2_df
from rev_ai import Transcript
from textgrids import TextGrid


class FluencyFeatureAnnotator:
    def __init__(self) -> None:
        self.fa = Wav2VecCTCFA()
        self.annotator = Annotator(process=["eos_detect", "pruning", "clause_detect"])
        self.extractor = UtteranceFluencyMeasureExtractor()

    def preprocess_for_turn(self, rev_transcript: Transcript) -> Tuple[pd.DataFrame, List[int]]:
        df_rev = transcript_2_df(rev_transcript)

        mask = [not(flag) for flag in df_rev["text"].str.contains("<.*?>", regex=True, na=True)]
        df_rev = df_rev[mask] # inaudible, laughter 等を除去

        filler_locations = [] # フィラーの word id のリスト
        idx = -1
        for i in df_rev.index:
            w = df_rev.at[i, "text"]
            t = df_rev.at[i, "type"]

            if t == "text":
                idx += 1
                if w.lower() in FILLER:
                    filler_locations.append(idx)
                if " " in w:
                    w = w.replace(" ", "_")
                    df_rev.at[i, "text"] = w

        df_text = df_rev[df_rev["type"] != "punct"].reset_index()
        df_text["text"] = df_text["text"].str.lower()

        return df_text, filler_locations

    def df_2_turn(self, df_rev: pd.DataFrame, filler_locations: List[int]) -> Turn:
        turn = Turn.from_DataFrame(df_rev, 0, word_col="text")

        disfluency_list = [DisfluencyEnum.FILLER for _ in filler_locations]
        turn.clauses[0].annotate_disfluency(filler_locations, disfluency_list)
        turn.reset_words()

        return turn

    def find_pauses(seflf, turn: Turn) -> List[dict]:
        pauses = []
        prev_clause_end = turn.start_time
        for clause in turn.clauses:
            if clause.start_time - prev_clause_end >= 0.25:
                p = {
                    "location": PauseLocation.CLAUSE_EXTERNAL,
                    "start_time": prev_clause_end,
                    "end_time": clause.start_time
                }
                pauses.append(p)

            prev_word_end = clause.start_time
            for word in clause.words:
                if word.start_time - prev_word_end >= 0.25:
                    p = {
                        "location": PauseLocation.CLAUSE_INTERNAL,
                        "start_time": prev_word_end,
                        "end_time": word.start_time
                    }
                    pauses.append(p)

                prev_word_end = word.end_time

            prev_clause_end = clause.end_time

        return pauses

    def wav_txt_file_path_generator(
        self,
        wav_file_path_list: List[Path],
        txt_file_path_list: List[Path]
    ) -> Generator[Tuple[Path, Path], None, None]:
        for wav_file_path in wav_file_path_list:
            for txt_file_path in txt_file_path_list:
                if txt_file_path.stem != wav_file_path.stem:
                    continue

                yield wav_file_path, txt_file_path
                break

    def annotate(
        self,
        wav_file_path_list: List[Path],
        txt_file_path_list: List[Path]
    ) -> Tuple[List[Turn], List[TextGrid]]:
        turn_list = []
        grid_list = []

        for file_path in self.wav_txt_file_path_generator(wav_file_path_list, txt_file_path_list):
            wav_file_path, txt_file_path = file_path

            with open(txt_file_path, "r") as f:
                transcript = f.readline()

            df_fa = self.fa.align(wav_file_path, transcript)
            rev_transcript = df_2_rev(df_fa)
            rev_transcript = Transcript.from_json(rev_transcript)

            df_transcript, filler_location = self.preprocess_for_turn(rev_transcript)
            turn = self.df_2_turn(df_transcript, filler_location)

            turn.ignore_disfluency()
            turn = self.annotator(turn=turn)
            turn.show_disfluency()

            pauses = self.find_pauses(turn)
            grid = self.annotator.to_textgrid(turn, pauses)

            turn_list.append(turn)
            grid_list.append(grid)

        return turn_list, grid_list

    def extract(self, turn_list: List[Turn], grid_list: List[TextGrid]) -> Tuple[List[List[float]], List[str]]:
        measure_list = []
        for turn, grid in zip(turn_list, grid_list):
            measures = self.extractor.extract_by_turn(turn, grid)
            measure_list.append(measures)

        measure_names = self.extractor.check_feature_names()

        return measure_list, measure_names

def save_turn(turn: Turn, save_path: Union[str, Path]) -> None:
    lines = []
    turn.show_disfluency()
    line = []
    for clause in turn.clauses:
        line.append(str(clause))

    lines.append(f"text_without_pruning: {' :: '.join(line)}\n\n")

    turn.ignore_disfluency()
    line = []
    for clause in turn.clauses:
        line.append(str(clause))
    lines.append(f"text_with_pruning: {' :: '.join(line)}")

    with open(save_path, "w") as f:
        f.writelines(lines)

def save_grid(grid: TextGrid, save_path: Union[str, Path]) -> None:
    if isinstance(save_path, str):
        textgrid_path = Path(save_path)
    elif isinstance(save_path, Path):
        textgrid_path = save_path.resolve()
    else:
        raise ValueError("textgrid_path must be path like object")

    grid.write(textgrid_path)

    grid = []
    with open(textgrid_path, "r") as f:
        for line in f.readlines():
            if "PointTier" in line:
                line = line.replace("PointTier", "TextTier")
            elif "xpos" in line:
                line = line.replace("xpos", "number")
            elif "__point__" in line:
                line = line.replace("text", "mark")
                line = line.replace("__point__", "")

            grid.append(line)

    with open(textgrid_path, "w") as f:
        f.writelines(grid)
