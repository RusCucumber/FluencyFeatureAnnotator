import json
from collections import deque
from pathlib import Path
from typing import List, Union

import numpy as np
from modules.common import RESOURCES_PATH
from modules.pipeline.utils.semantic_graph import SemanticGraph
from spacy.tokens.span import Span
from stanza.models.common.doc import Sentence

MARK_2_ID_PATH = RESOURCES_PATH / "dependency_parser/mark_2_id.json"

class DependencyTree:
    def __init__(
        self,
        words: List[str],
        semantic_graph: SemanticGraph,
        mark_2_id: Union[dict, str, Path] =MARK_2_ID_PATH
    ) -> 'DependencyTree':
        assert isinstance(words, list), "words must be list of str"

        assert isinstance(semantic_graph, SemanticGraph), f"semantic_graph must be {repr(SemanticGraph)}"

        assert any(
            [isinstance(mark_2_id, dict), isinstance(mark_2_id, str), isinstance(mark_2_id, Path)]
            ), "mark_2_id must be dict, str or Path"

        assert len(words) == len(semantic_graph), "the amount of words and graph's node are not same"

        if isinstance(mark_2_id, str) or isinstance(mark_2_id, Path):
            mark_2_id = self.load_mark_2_id(mark_2_id)

        self.__words = words
        self.__tree = semantic_graph
        self.__id_2_mark = self.generate_id_2_mark(mark_2_id)
        self.__mark_2_id = mark_2_id

    @classmethod
    def from_SpacySentence(cls, sentence: Span, mark_2_id: Union[dict, str, Path]=MARK_2_ID_PATH) -> 'DependencyTree':
        assert isinstance(sentence, Span), f"sentence must be {Span}"

        words = ["root"] + [word.text for word in sentence]

        mark_2_id = cls.load_mark_2_id(cls, mark_2_id)
        semantic_graph = SemanticGraph.from_SpacySentence(sentence, mark_2_id)

        return cls(words, semantic_graph, mark_2_id=mark_2_id)

    @classmethod
    def from_StanzaSentence(
        cls,
        sentence: Sentence,
        mark_2_id: Union[dict, str, Path]=MARK_2_ID_PATH
    ) -> 'DependencyTree':
        assert isinstance(sentence, Sentence), f"sentence must be {Sentence}"

        words = ["root"] + [word.text for word in sentence.words]

        mark_2_id = cls.load_mark_2_id(cls, mark_2_id)
        semantic_graph = SemanticGraph.from_StanzaSentence(sentence, mark_2_id)

        return cls(words, semantic_graph, mark_2_id=mark_2_id)

    def load_mark_2_id(self, mark_2_id_path: Union[str, Path]) -> dict:
        path = Path(mark_2_id_path).resolve()

        with open(path) as j:
            mark_2_id = json.load(j)

        return mark_2_id

    def generate_id_2_mark(self, mark_2_id: dict) -> dict:
        assert isinstance(mark_2_id, dict), "mark_2_id must be dict"

        id_2_mark = {}
        for val, key in mark_2_id.items():
            id_2_mark[key] = val

        return id_2_mark

    def display(self) -> None:
        word_ids = self.get_word_id()
        for i in word_ids:
            print(f"id: {i:3d}", end="\t")
            print(f"word: {self.__words[i]}", end="\t\n")
            for j, dep_id in enumerate(self.__tree[i,:]):
                if dep_id == 0:
                    continue

                print(f"\t|--{self.__id_2_mark[dep_id]}--> {self.__words[j]}")
            print()

    def get_word_id(self) -> List[int]:
        word_id = []
        word_id += np.where(np.sum(self.__tree, axis=0))[0].tolist()
        word_id += np.where(np.sum(self.__tree, axis=1))[0].tolist()
        word_id = list(set(word_id))

        return sorted(word_id)

    def separate(self, word_id: int) -> 'DependencyTree':
        separated_graph = np.zeros(self.__tree.shape)
        queue = deque([word_id])

        root = False
        while len(queue) > 0:
            i = queue.popleft()

            # 対象のノードの子をコピーする
            separated_graph[i, :] = self.__tree[i, :]

            # 対象のノードの親を探す
            j = self.__find_parent_by_i(i)

            if not root:
                root = j # 親のみになると，ゼロ行列になるため，最初の親のみ保持

            # 対象のノードを木から切り離す
            self.__tree[j, i] = 0

            children_idx = self.__find_children_by_i(i)
            for idx in children_idx:
                queue.append(idx)

        if np.sum(separated_graph) == 0:
            separated_graph[word_id, word_id] = self.__mark_2_id["root"]

        separated_graph = SemanticGraph.from_list(separated_graph)

        if np.sum(self.__tree) == 0:
            self.__tree[root, root] = self.__mark_2_id["root"]

        return DependencyTree(self.__words, separated_graph)

    def is_dependency_exist(self, deprel: str) -> List[int]:
        dep_id = self.__mark_2_id[deprel]
        word_ids = np.where(np.sum(self.__tree, axis=0) == dep_id)[0]

        return word_ids.tolist()

    def are_dependencies_exist(self, deprel_list: List[str]) -> List[int]:
        if isinstance(deprel_list, str):
            deprel_list = [deprel_list]

        word_ids = []
        for deprel in deprel_list:
            word_ids += self.is_dependency_exist(deprel)

        return sorted(word_ids)

    def __find_parent_by_i(self, i: int) -> int:
        j = np.where(self.__tree[:, i] > 0)[0]

        assert len(j) == 1, "Find 2 parents"

        return j[0]

    def __find_children_by_i(self, i: int) -> int:
        children_candidates = self.__tree[i, :]
        assert children_candidates[i] == 0 or children_candidates[i] == self.__mark_2_id["root"],\
              "Find loop on the graph"

        children_id = np.where(children_candidates > 0)[0]

        return children_id
