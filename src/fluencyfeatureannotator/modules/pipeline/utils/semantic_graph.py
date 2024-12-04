from copy import deepcopy
from typing import Union

import numpy as np
from spacy.tokens.span import Span
from stanza.models.common.doc import Sentence


class SemanticGraph:
    def __init__(self, n_node: int) -> 'SemanticGraph':
        self.__graph = np.zeros((n_node, n_node), dtype=np.int32)

    @property
    def graph(self):
        return deepcopy(self.__graph)

    @graph.setter
    def graph(self, graph: Union[list, np.ndarray]) -> None:
        if type(graph) is list:
            graph = np.array(graph, dtype=np.int32)
        elif type(graph) is  np.ndarray:
            graph = graph.astype(np.int32)
        else:
            raise TypeError("graph must be list like object")

        if self.__graph.shape != graph.shape:
            raise TypeError("inputed graph's shape is different")

        self.__graph = graph

    @property
    def shape(self) -> tuple:
        return self.__graph.shape

    def __repr__(self) -> str:
        return "SemanticGraph"

    def __str__(self) -> str:
        return f"{self.__graph}"

    def __add__(self, graph: 'SemanticGraph') -> 'SemanticGraph':
        if type(graph) is not SemanticGraph:
            raise TypeError("semantic graph only can calculate with semantic graph")

        return self.__graph + graph.graph

    def __sub__(self, graph: 'SemanticGraph') -> 'SemanticGraph':
        if type(graph) is not SemanticGraph:
            raise TypeError("semantic graph only can calculate with semantic graph")

        return self.__graph - graph.graph

    def __iadd__(self, graph: 'SemanticGraph') -> 'SemanticGraph':
        if type(graph) is not SemanticGraph:
            raise TypeError("semantic graph only can calculate with semantic graph")

        self.__graph += graph.graph

    def __isub__(self, graph: 'SemanticGraph') -> 'SemanticGraph':
        if type(graph) is not SemanticGraph:
            raise TypeError("semantic graph only can calculate with semantic graph")

        self.__graph -= graph.graph

    def __eq__(self, value: int) -> np.ndarray:
        if type(value) not in (int, float):
            raise TypeError("semantic graph only can compare with number")

        return self.__graph == value

    def __ne__(self, value: int) -> np.ndarray:
        if type(value) not in (int, float):
            raise TypeError("semantic graph only can compare with number")

        return self.__graph != value

    def __lt__(self, value: int) -> np.ndarray:
        if type(value) not in (int, float):
            raise TypeError("semantic graph only can compare with number")

        return self.__graph < value

    def __le__(self, value: int) -> np.ndarray:
        if type(value) not in (int, float):
            raise TypeError("semantic graph only can compare with number")

        return self.__graph <= value

    def __gt__(self, value: int) -> np.ndarray:
        if type(value) not in (int, float):
            raise TypeError("semantic graph only can compare with number")

        return self.__graph > value

    def __ge__(self, value: int) -> np.ndarray:
        if type(value) not in (int, float):
            raise TypeError("semantic graph only can compare with number")

        return self.__graph >= value

    def __len__(self) -> int:
        return len(self.__graph)

    def __getitem__(self, key: int) -> int:
        if type(key) not in (int, tuple, slice):
            raise TypeError("semantic graph indices must be int, tuple or slice")

        return self.__graph[key]

    def __setitem__(self, key: int, val: int) -> None:
        if type(key) not in (int, tuple, slice):
            raise TypeError("semantic graph indices must be int, tuple or slice")

        self.__graph[key] = val

    def __delitem__(self, key: int) -> None:
        if type(key) not in (int, tuple, slice):
            raise TypeError("semantic graph indices must be int, tuple or slice")

        self.__graph[key] = 0

    @classmethod
    def from_StanzaSentence(cls, sentence: Sentence, mark_2_id: dict) -> 'SemanticGraph':
        assert isinstance(sentence, Sentence), f"sentence must be {Sentence}"

        assert isinstance(mark_2_id, dict), "mark_2_id must be dict"

        n_node = len(sentence.words) + 1
        dep_graph = cls(n_node)

        for word in sentence.words:
            i = word.head
            j = word.id

            dep_graph[i, j] = mark_2_id[word.deprel]

        return dep_graph

    @classmethod
    def from_SpacySentence(cls, sentence: Span, mark_2_id: dict) -> 'SemanticGraph':
        assert isinstance(sentence, Span), f"sentence must be {Span}"
        assert isinstance(mark_2_id, dict), "mark_2_id must be dict"

        n_node = len(sentence) + 1
        dep_graph = cls(n_node)

        idx_calibrator = None
        for token in sentence:
            if idx_calibrator is None:
                idx_calibrator = token.i - 1

            i = token.head.i - idx_calibrator
            j = token.i - idx_calibrator
            if i == j:
                i = 0

            dep_graph[i, j] = mark_2_id[token.dep_]

        return dep_graph

    @classmethod
    def from_list(cls, list_like_obj: Union[list, np.ndarray]) -> 'SemanticGraph':
        if type(list_like_obj) is list:
            list_like_obj = np.array(list_like_obj, dtype=np.int32)
        elif type(list_like_obj) is np.ndarray:
            list_like_obj = list_like_obj.astype(np.int32)
        else:
            raise ValueError("intput must be list like object (e.g. list, ndarray)")

        n_node = list_like_obj.shape[0]

        graph = cls(n_node)
        graph.graph = list_like_obj

        return graph

