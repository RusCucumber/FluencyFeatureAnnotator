import re
import sys
from pathlib import Path

CWD = Path(__file__).parent
sys.path.append(
    str(CWD / "utils")
)

import pandas as pd
from text.cleaners import collapse_whitespace, english_cleaners

INAUDIBLE_TAGS = "<inaudible>"
PUNCTUATIONS = [".", ",", "?", "!"]
TAGS = [
    r"\(.*?\)",
    r"\[.*?\]",
    r"\<.*?\>",
    r"\{.*?\}"
]
WORD_CONNECTORS = ["-", ":"]

def text_preprocess(text_raw: str) -> str:
    # 1. apply english cleaners
    text_trans = english_cleaners(text_raw)

    # 2. <inaudible> tags to *
    text_trans = text_trans.replace(INAUDIBLE_TAGS, "*")

    # 3. remove other tags
    for tag in TAGS:
        text_trans = re.sub(tag, " ", text_trans)

    # 4. remove punctuations
    for punct in PUNCTUATIONS:
        text_trans = text_trans.replace(punct, " ")

    # 5. remove word connectors (e.g., "-", ":")
    for word_connector in WORD_CONNECTORS:
        text_trans = text_trans.replace(word_connector, " ")

    # 6. transform % to "percent(s)"
    if " one% " in text_trans:
        text_trans = text_trans.replace(" one% ", " one percent ")
    if "%" in text_trans:
        text_trans = text_trans.replace("%", " percents")

    # 7. transform & to "and"
    and_pattern = r"(\w)&(\w)"
    text_trans = re.sub(
        and_pattern, r"\1 & \2", text_trans
    )

    if "&" in text_trans:
        text_trans = text_trans.replace("&", "and")

    # 8. remove unnecessary spaces
    text_trans = collapse_whitespace(text_trans)
    text_trans = text_trans.strip()

    return text_trans

def text_postprocess(text_raw: str, df_timestamp: pd.DataFrame) -> pd.DataFrame:
    # 1. apply english cleaners
    text_trans = english_cleaners(text_raw)

    # 2. <inaudible> tags to [INAUDIBLE]
    text_trans = text_trans.replace(INAUDIBLE_TAGS, "#INAUDIBLE#")

    # 3. remove other tags
    for tag in TAGS:
        text_trans = re.sub(tag, " ", text_trans)

    # 4. replace [INAUDIBLE] to <inaudible>
    text_trans = text_trans.replace("#INAUDIBLE#", INAUDIBLE_TAGS)

    # 5. remove punctuations
    for punct in PUNCTUATIONS:
        text_trans = text_trans.replace(punct, " ")

    # 6. add word connector sign (e.g., "-", ":")
    for word_connector in WORD_CONNECTORS:
        text_trans = text_trans.replace(word_connector, f" {word_connector}")

    # 7. transform % to "percent(s)"
    if " one% " in text_trans:
        text_trans = text_trans.replace(" one% ", " one percent ")
    if "%" in text_trans:
        text_trans = text_trans.replace("%", " percents")

    # 8. transform & to "and"
    and_pattern = r"(\w)&(\w)"
    text_trans = re.sub(
        and_pattern, r"\1 & \2", text_trans
    )

    if "&" in text_trans:
        text_trans = text_trans.replace("&", "and")

    # 9. remove unnecessary spaces
    text_trans = collapse_whitespace(text_trans)
    text_trans = text_trans.strip()

    original_words = text_trans.split(" ")
    if len(original_words) != len(df_timestamp):
        raise RuntimeError(
            f"\nThe length of original (N={len(original_words)}) and FA texts (N={len(df_fa)} is NOT equal)"
        )

    data = []
    for idx, original_word in enumerate(original_words):
        start_time = df_timestamp.at[idx, "start_time"]
        end_time = df_timestamp.at[idx, "end_time"]

        # - や : で結合された単語の処理
        if original_word[0] in WORD_CONNECTORS:
            prev_word_block = data[-1]
            prev_word_block["word"] += original_word
            prev_word_block["end_time"] = end_time

            data[-1] = prev_word_block
            continue

        row = {
            "word": original_word,
            "start_time": start_time,
            "end_time": end_time
        }
        data.append(row)

    df_timestamp_processed = pd.DataFrame.from_dict(data)
    return df_timestamp_processed
