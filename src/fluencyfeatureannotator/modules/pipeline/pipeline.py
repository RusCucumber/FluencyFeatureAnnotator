from math import isclose
from pathlib import Path
from typing import List, Optional, Tuple, Union

import textgrids
from modules.common import DisfluencyWord, Turn
from modules.pipeline.asr import ASR
from modules.pipeline.dependency_parser import SpacyClauseDetector, StanzaClauseDetector
from modules.pipeline.pause_location_detector import PauseLocationDetector
from modules.pipeline.pruning import Pruning
from modules.pipeline.sentence_end_detector import SentenceEndDetector


class Annotator:
    def __init__(
            self,
            process: List[str] =["asr", "eos_detect", "pruning", "clause_detect", "pause_location"],
            frame_rate: int =16000,
            model: str ="webrtc",
            vad_config: dict ={},
            disfluency_detector: str ="roberta",
            dependency_parser: str ="spacy",
            device: str ="cpu"
    ):
        self.__process = process
        self.__modules = self.__set_process(
            process,
            frame_rate,
            model,
            vad_config,
            disfluency_detector,
            dependency_parser,
            device
        )

    # 出力方法の決定処理
    def __call__(
        self,
        wav_path: Optional[Union[str, Path]] =None,
        turn: Optional[Turn] =None
    ) -> Union[Turn, Tuple[Turn, List[dict]]]:
        return self.annotate(wav_path, turn)

    def transcribe(self, wav_path: Union[str, Path]) -> Turn:
        assert "asr" in self.__process, "Annotation pipeline not includes <ASR>"

        asr = self.__modules["asr"]
        return asr(wav_path)

    def detect_sentence_end(self, turn: Turn) -> Turn:
        assert "eos_detect" in self.__process, "Annotation pipeline not includes <sentence end detector>"

        sed = self.__modules["eos_detect"]
        return sed(turn)

    def pruning(self, turn: Turn) -> Turn:
        assert "pruning" in self.__process, "Annotation pipeline not includes <pruning>"

        pruning = self.__modules["pruning"]
        return pruning(turn)

    def detect_clause_end(self, turn: Turn) -> Turn:
        assert "clause_detect" in self.__process, "Annotation pipeline not includes <clause detector>"

        cd = self.__modules["clause_detect"]
        return cd(turn)

    def detect_pause_location(self, wav_path: Union[str, Path], turn: Turn) -> List[dict]:
        assert "pause_location" in self.__process, "Annotation pipeline not includes <pause location detector>"

        pld = self.__modules["pause_location"]
        return pld(wav_path, turn)

    def annotate(
        self,
        wav_path: Optional[Union[str, Path]] =None,
        turn: Optional[Turn] =None
    ) -> Union[Turn, Tuple[Turn, List[dict]]]:
        for name, mod in self.__modules.items():
            if name == "asr":
                turn = mod(wav_path)
            elif name == "pause_location":
                pause_locatin = mod(wav_path, turn)

                return turn, pause_locatin
            else:
                turn = mod(turn)

        return turn

    def to_textgrid( #TODO: too complex
        self,
        turn: Turn,
        pause_location: List[dict],
        save_path: Union[str, Path] =None
    ) -> textgrids.TextGrid:
        save_textgrid = True

        if save_path is None:
            save_textgrid = False
            save_path = ""
        elif isinstance(save_path, Path):
            save_path = str(save_path)
        assert isinstance(save_path, str), "save path must be string"

        xmin = 0.0

        if len(pause_location) == 0:
            xmax = turn.end_time
        else:
            xmax = max(turn.end_time, pause_location[-1]["end_time"])

        grid = textgrids.TextGrid()
        grid.xmin = xmin
        grid.xmax = xmax

        # 書き起こしの interval tier
        turn.ignore_disfluency()

        transcripts = []
        prev_xmax = xmin
        for clause in turn.clauses:
            if not isclose(prev_xmax, clause.start_time):
                if clause.start_time > prev_xmax:
                    interval = textgrids.Interval(text="", xmin=prev_xmax, xmax=clause.start_time)
                    transcripts.append(interval)

            interval = textgrids.Interval(text=str(clause), xmin=clause.start_time, xmax=clause.end_time)
            transcripts.append(interval)

            prev_xmax = clause.end_time

        if prev_xmax < xmax:
            interval = textgrids.Interval(text="", xmin=prev_xmax, xmax=xmax)
            transcripts.append(interval)

        transcripts = textgrids.Tier(data=transcripts)

        # ポーズの interval tier
        pauses = []
        prev_xmax = xmin
        for pause in pause_location:
            if prev_xmax != pause["start_time"]:
                interval = textgrids.Interval(text="", xmin=prev_xmax, xmax=pause["start_time"])
                pauses.append(interval)

            pl = pause["location"]
            if pl.name == "CLAUSE_INTERNAL":
                pl = "CI"
            elif pl.name == "CLAUSE_EXTERNAL":
                pl = "CE"
            else:
                pl = pl.name

            interval = textgrids.Interval(text=pl, xmin=pause["start_time"], xmax=pause["end_time"])
            pauses.append(interval)

            prev_xmax = pause["end_time"]

        if prev_xmax < xmax:
            interval = textgrids.Interval(text="", xmin=prev_xmax, xmax=xmax)
            pauses.append(interval)

        pauses = textgrids.Tier(data=pauses)

        # 言い淀みの point tier
        turn.show_disfluency()

        disflu = []
        rep = []
        Self = []
        false = []
        filler = []
        for word in turn.words:
            if isinstance(word, DisfluencyWord):
                kind = word.dismodules.name
                start = word.start_time

                point = textgrids.Point(f"__point__{word.text}", start)

                if kind == "FILLER":
                    filler.append(point)
                    continue
                else:
                    disflu.append(point)

                if kind == "REPETITION":
                    rep.append(point)
                elif kind == "SELF_REPAIR":
                    Self.append(point)
                elif kind == "FALSE_START":
                    false.append(point)

        disflu = textgrids.Tier(data=disflu, point_tier=True)
        rep = textgrids.Tier(data=rep, point_tier=True)
        Self = textgrids.Tier(data=Self, point_tier=True)
        false = textgrids.Tier(data=false, point_tier=True)
        filler = textgrids.Tier(data=filler, point_tier=True)

        grid["transcript"] = transcripts
        grid["pause"] = pauses
        grid["filler"] = filler
        grid["disflu"] = disflu
        grid["rep"] = rep
        grid["self"] = Self
        grid["false"] = false

        if save_textgrid:
            self.__write_textgrid(grid, save_path)

        return grid

    # TODO: csv 出力処理の作成
    def to_csv(self, turn, pause_location, save_path):
        pass

    # TODO: グラフの出力処理の作成
    def to_graph(self, turn, pause_location, save_path):
        pass

    def __set_process(
            self,
            process,
            frame_rate,
            model,
            vad_config,
            disfluency_detector,
            dependency_parser,
            device
    ):
        modules = {}
        if "asr" in process:
            modules["asr"] = ASR(frame_rate=frame_rate)

        if "eos_detect" in process:
            modules["eos_detect"] = SentenceEndDetector(device=device)

        if "pruning" in process:
            modules["pruning"] = Pruning(device=device, model=disfluency_detector)

        if "clause_detect" in process:
            if dependency_parser == "spacy":
                modules["clause_detect"] = SpacyClauseDetector(device=device)
            elif dependency_parser == "stanza":
                modules["clause_detect"] = StanzaClauseDetector(device=device)

        if "pause_location" in process:
            modules["pause_location"] = PauseLocationDetector(model=model, vad_config=vad_config)

        return modules

    def __write_textgrid(self, textgrid: textgrids.TextGrid, textgrid_path: Union[str, Path]) -> None:
        assert isinstance(textgrid, textgrids.TextGrid), f"textgrid must be {textgrids.TextGrid}"

        if isinstance(textgrid_path, str):
            textgrid_path = Path(textgrid_path)
        elif isinstance(textgrid_path, Path):
            textgrid_path = textgrid_path.resolve()
        else:
            raise ValueError("textgrid_path must be path like object")

        textgrid.write(textgrid_path)

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
