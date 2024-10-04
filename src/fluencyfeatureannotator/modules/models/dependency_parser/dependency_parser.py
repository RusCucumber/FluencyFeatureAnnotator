from pathlib import Path
from typing import Union

import spacy
import stanza
from modules.common import RESOURCES_PATH


class StanzaDependencyParser:
    def __init__(
            self,
            stanza_dir: Union[str, Path] =RESOURCES_PATH,
            device: str ="cpu"
        ) -> 'StanzaDependencyParser':
        if not RESOURCES_PATH.exists():
            raise FileNotFoundError(f"Not found stanza_dir: {str(RESOURCES_PATH)}")

        if isinstance(stanza_dir, str):
            stanza_dir = Path(stanza_dir).resolve()
        elif isinstance(stanza_dir, Path):
            stanza_dir = stanza_dir.resolve()
        else:
            raise ValueError("stanza_dir must be path like object")

        use_gpu = True
        if device == "cpu":
            use_gpu = False

        stanza_resources_dir = (stanza_dir / "stanza_resources").resolve()
        self.__nlp = stanza.Pipeline(
            "en",
            dir=str(stanza_resources_dir),
            processors='tokenize,mwt,pos,lemma,depparse',
            use_gpu=use_gpu
        )

    def __call__(self, text: str) -> stanza.Document:
        return self.predict(text)

    def predict(self, text: str) -> stanza.Document:
        doc = self.__nlp(text)
        return doc

class SpacyDependencyParser:
    def __init__(
            self,
            model_name: str ="en_ud_L1L2e_combined_trf"
        ) -> 'SpacyDependencyParser':
        self.__nlp = spacy.load(model_name)

    def __call__(self, text: str) -> spacy.tokens.doc.Doc:
        return self.predict(text)

    def predict(self, text: str) -> spacy.tokens.doc.Doc:
        doc = self.__nlp(text)
        return doc
