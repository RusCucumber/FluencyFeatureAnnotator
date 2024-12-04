from typing import Dict, List

import pandas as pd


def generate_word_block(word: str, start_time: float, end_time: float) -> List[dict]:
    word_block_list = []

    if not isinstance(word, str):
        word = str(word)

    words = word.split(" ")
    if len(words) == 1:
        word = {
            "type": "text",
            "value": word,
            "ts": start_time,
            "end_ts": end_time,
            "confidence": 1.0
        }
        word_block_list.append(word)
        return word_block_list

    delta = (end_time - start_time) / len(words)
    for t, w in enumerate(words):
        word = {
            "type": "text",
            "value": w,
            "ts": start_time + (delta * t),
            "end_ts": start_time + (delta * (t + 1)),
            "confidence": 1.0
        }
        word_block_list.append(word)
        punct = {
            "type": "punct",
            "value": " "
        }
        word_block_list.append(punct)

    return word_block_list[:-1]

def generate_rev_element(df_fa: pd.DataFrame) -> List[Dict[str, str]]:
    element = []
    for idx in df_fa.index:
        word = df_fa.at[idx, "word"]

        if word == "":
            continue

        start_time = df_fa.at[idx, "start_time"]
        end_time = df_fa.at[idx, "end_time"]

        word_block_list = generate_word_block(word, start_time, end_time)
        element += word_block_list
        punct = {
            "type": "punct",
            "value": " "
        }
        element.append(punct)

    element[-1]["value"] = "."

    return element

def df_2_rev(df_fa: pd.DataFrame) -> Dict[str, List[dict]]:
    element = generate_rev_element(df_fa)

    rev_json = {
        "monologues": [
            {
                "speaker": 0,
                "elements": element
            }
        ]
    }

    return rev_json

