from typing import List

import numpy as np
import syllables
from modules.common import Turn
from textgrids import TextGrid


class UtteranceFluencyMeasureExtractor:
    def __init__(
        self, sr:bool =True, mcpr:bool =True, ecpr:bool =True, mcpd:bool =True,
        ecpd:bool =True, fp_ratio:bool =True, dys_ratio:bool =True, dys_rate:bool =True,
        rep:bool =True, rpr:bool =True, fs:bool =True, rf:bool =True, ar:bool =True, ptr:bool =True,
        mlr:bool =True, ar_w:bool =True, sr_w:bool =True, mlr_w:bool =True, mpd:bool =True,
        df_col_word: str ="word", df_col_pause: str ="PAUSE", df_col_pdur: str ="p_dur",
        mc_tag: str ="CI", ec_tag: str ="CE"
    ):
        self.config = {
            "speech_rate": sr,
            "mid_clause_pause_ratio": mcpr,
            "end_clause_pause_ratio": ecpr,
            "mid_clause_p-dur": mcpd,
            "end_clause_p-dur": ecpd,
            "filled_pause_ratio": fp_ratio,
            "dysfluency_ratio": dys_ratio,
            "dysfluency_rate": dys_rate,
            "repetition_ratio": rep,
            "self_repair_ratio": rpr,
            "false_start_ratio": fs,
            "repair_false_ratio": rf,
            "articulation_rate": ar,
            "phonation_time_rate": ptr,
            "mean_length_of_run": mlr,
            "articulation_rate_word": ar_w,
            "speech_rate_word": sr_w,
            "mean_length_of_run_word": mlr_w,
            "mean_pause_duration": mpd
        }

        self.__calc = {
            "speech_rate": self.calc_speech_rate,
            "mid_clause_pause_ratio": self.calc_mid_clause_pause_ratio,
            "end_clause_pause_ratio": self.calc_end_clause_pause_ratio,
            "mid_clause_p-dur": self.calc_mid_clause_pause_dur,
            "end_clause_p-dur": self.calc_end_clause_pause_dur,
            "filled_pause_ratio": self.calc_filled_pause_ration,
            "dysfluency_ratio": self.calc_dysfluency_ratio,
            "dysfluency_rate": self.calc_dysfluency_rate,
            "repetition_ratio": self.calc_repetition_ratio,
            "self_repair_ratio": self.calc_self_repair_ratio,
            "false_start_ratio": self.calc_false_start_ratio,
            "repair_false_ratio": self.calc_repair_false_ratio,
            "articulation_rate": self.calc_articulation_rate,
            "phonation_time_rate": self.calc_phonation_time_rate,
            "mean_length_of_run": self.calc_mean_length_of_run,
            "articulation_rate_word": self.calc_articulation_rate_word,
            "speech_rate_word": self.calc_speech_rate_word,
            "mean_length_of_run_word": self.calc_mean_length_of_run_word,
            "mean_pause_duration": self.calc_mean_pause_duration
        }

        self.__word = df_col_word
        self.__pause = df_col_pause
        self.__p_dur = df_col_pdur
        self.__mc_tag = mc_tag
        self.__ec_tag = ec_tag

    def extract_by_parameters(self, params: dict) -> List[float]:
        feature_list = []
        for feature, is_calc in self.config.items():
            if is_calc:
                feature_list.append(self.__calc[feature](params))

        return feature_list

    def extract_by_turn(self, turn: Turn, grid: TextGrid, pruning:bool =True) -> List[float]:
        params = self.calc_parameters(turn, grid, pruning=pruning)

        return self.extract_by_parameters(params)

    def join_parameters(self, params_list: List[dict]) -> dict:
        joined_params = {
            "n_word": 0,
            "dur": 0,
            "syl": np.array([]),
            "mc": np.array([]),
            "ec": np.array([]),
            "repetition": 0,
            "self_repair": 0,
            "false_start": 0,
            "repair_false": 0
        }

        for params in params_list:
            joined_params["n_word"] += params["n_word"]
            joined_params["dur"] += params["dur"]
            joined_params["syl"] = np.hstack([joined_params["syl"], params["syl"]])
            joined_params["mc"] = np.hstack([joined_params["mc"], params["mc"]])
            joined_params["ec"] = np.hstack([joined_params["ec"], params["ec"]])
            joined_params["repetition"] += params["repetition"]
            joined_params["self_repair"] += params["self_repair"]
            joined_params["false_start"] += params["false_start"]
            joined_params["repair_false"] += params["repair_false"]

        return joined_params

    # パラメータ計算処理
    def calc_parameters(self, turn: Turn, grid: TextGrid, pruning: bool =True) -> dict: #TODO: too complex
        n_word = 0
        dur = turn.end_time - turn.start_time
        syl = []
        mc = []
        ec = []
        repepetion = 0
        self_repair = 0
        false_start = 0
        filled = 0

        keys = list(grid.keys())

        if pruning:
            turn.ignore_disfluency() # pruning
        else:
            turn.show_disfluency() # w/o pruning

        for word in turn.words:
            syl.append(syllables.estimate(word.text))
            n_word += 1

        # TODO: grid[keys[...]] の箇所を hardcoding から修正
        # TODO: pause のしきい値を設定可能に
        for pause in grid[keys[1]]:
            p_dur = pause.xmax - pause.xmin
            if p_dur <= 0.25:
            # if p_dur <= 0.15:
                continue

            if pause.text == self.__mc_tag:
                mc.append(p_dur)
            elif pause.text == self.__ec_tag:
                ec.append(p_dur)

        for _ in grid[keys[-3]]:
            repepetion += 1

        for _ in grid[keys[-2]]:
            self_repair += 1

        for _ in grid[keys[-1]]:
            false_start += 1

        for _ in grid[keys[2]]:
            filled += 1

        params = {
            "n_word": n_word,
            "dur": dur,
            "syl": np.array(syl),
            "mc": np.array(mc),
            "ec": np.array(ec),
            "repetition": repepetion,
            "self_repair": self_repair,
            "false_start": false_start,
            "repair_false": self_repair + false_start,
            "filled": filled
        }

        turn.show_disfluency()

        return params

    def calc_speech_rate(self, params: dict) -> float:
        n_syl = params["syl"].sum()
        return self.__divide_safe(n_syl, params["dur"])

    def calc_mid_clause_pause_ratio(self, params: dict) -> float:
        total_mc_appearance = len(params["mc"])
        n_syl = params["syl"].sum()
        return self.__divide_safe(total_mc_appearance, n_syl)

    def calc_end_clause_pause_ratio(self, params: dict) -> float:
        total_ec_appearance = len(params["ec"])
        n_syl = params["syl"].sum()
        return self.__divide_safe(total_ec_appearance, n_syl)

    def calc_mid_clause_pause_dur(self, params: dict) -> float:
        nonzero_mc = params["mc"][np.nonzero(params["mc"])]

        if len(nonzero_mc) == 0:
            return 0

        return nonzero_mc.mean()

    def calc_end_clause_pause_dur(self, params: dict) -> float:
        nonzero_ec = params["ec"][np.nonzero(params["ec"])]

        if len(nonzero_ec) == 0:
            return 0

        return nonzero_ec.mean()

    def calc_filled_pause_ration(self, params: dict) -> float:
        n_syl = params["syl"].sum()

        return self.__divide_safe(params["filled"], n_syl)

    def calc_dysfluency_ratio(self, params: dict) -> float:
        dysfluency_keys = ["repetition", "self_repair", "false_start"]
        total_dysfluency = 0
        for key in dysfluency_keys:
            total_dysfluency += params[key]

        n_syl = params["syl"].sum()

        return self.__divide_safe(total_dysfluency, n_syl)

    def calc_dysfluency_rate(self, params: dict) -> float:
        dysfluency_keys = ["repetition", "self_repair", "false_start"]
        total_dysfluency = 0
        for key in dysfluency_keys:
            total_dysfluency += params[key]

        return self.__divide_safe(total_dysfluency, params["dur"])

    def calc_repetition_ratio(self, params: dict) -> float:
        n_syl = params["syl"].sum()

        return self.__divide_safe(params["repetition"], n_syl)

    def calc_self_repair_ratio(self, params: dict) -> float:
        n_syl = params["syl"].sum()

        return self.__divide_safe(params["self_repair"], n_syl)

    def calc_false_start_ratio(self, params: dict) -> float:
        n_syl = params["syl"].sum()

        return self.__divide_safe(params["false_start"], n_syl)

    def calc_repair_false_ratio(self, params: dict) -> float:
        n_syl = params["syl"].sum()

        return self.__divide_safe(params["repair_false"], n_syl)

    def calc_articulation_rate(self, params: dict) -> float:
        n_syl = params["syl"].sum()
        phonation_time = params["dur"] - params["mc"].sum() - params["ec"].sum()
        return self.__divide_safe(n_syl, phonation_time)

    def calc_phonation_time_rate(self, params: dict) -> float:
        phonation_time = params["dur"] - params["mc"].sum() - params["ec"].sum()
        return self.__divide_safe(phonation_time, params["dur"])

    def calc_mean_length_of_run(self, params: dict) -> float:
        n_pause = len(params["mc"]) + len(params["ec"])
        n_syl = params["syl"].sum()

        return self.__divide_safe(n_syl, n_pause + 1)

    def calc_articulation_rate_word(self, params: dict) -> float:
        phonation_time = params["dur"] - params["mc"].sum() - params["ec"].sum()
        return self.__divide_safe(params["n_word"], phonation_time)

    def calc_speech_rate_word(self, params: dict) -> float:
        return self.__divide_safe(params["n_word"], params["dur"])

    def calc_mean_length_of_run_word(self, params: dict) -> float:
        n_pause = len(params["mc"]) + len(params["ec"])

        return self.__divide_safe(params["n_word"], n_pause + 1)

    def calc_mean_pause_duration(self, params: dict) -> float:
        pause = params["mc"].sum() + params["ec"].sum()

        if pause == 0:
            return 0

        n_pause = len(params["mc"]) + len(params["ec"])

        return self.__divide_safe(pause, n_pause)

    def check_feature_names(self, show: bool =False) -> List[str]:
        feature_names = []
        for feature, is_use in self.config.items():
            if is_use:
                feature_names.append(feature)

        if show:
            print("Calculate following features...")
            for i, name in enumerate(feature_names):
                print(f"feature {i:2d}: {name}")
            print("", flush =True)

        return feature_names

    def __divide_safe(self, x: float, y: float) -> float:
        if y == 0:
            return np.nan
        return x / y

