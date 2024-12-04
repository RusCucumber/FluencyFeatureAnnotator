from collections import deque
from typing import List

from modules.common import Turn
from modules.models import SpacyDependencyParser, StanzaDependencyParser
from modules.pipeline.utils.base import ModuleBase
from modules.pipeline.utils.dependency_tree import DependencyTree
from spacy.tokens import Doc
from stanza import Document

MARK_2_ID_PATH = "DAF/resources/dependency_parser/mark_2_id.json"
SUBORDINATE_CLAUSE_MARKES = ["csubj", "csubj:pass", "ccomp", "xcomp", "advcl", "acl", "acl:relcl"]

# TODO: Spacy 版で class を全て書き換え
class SpacyClauseDetector(ModuleBase):
    def __init__(
        self,
        device: str ="cpu",
        subordinate_clause_marks: List[str] =SUBORDINATE_CLAUSE_MARKES
    ) -> 'SpacyClauseDetector':
        assert isinstance(subordinate_clause_marks, list), "subordinate_clause_marks must be list of str"

        super().__init__()
        self._model = SpacyDependencyParser()
        self.__sc_marks = subordinate_clause_marks

    def predict(self, turn: Turn) -> Turn:
        self._turn_type_checker(turn)

        turn.ignore_disfluency()
        text = ""
        for sentence in turn.clauses:
            text += (sentence.text + ". ")

        doc = self._model.predict(text)
        doc_tree = self.generate_dependency_tree(doc) # sentence ごとの dep tree のリスト
        id_calibrator = self.generate_id_calibrator(doc) # word id と doc id の補正
        doc_clause_ids = self.estimate_subordinate_clause_ids(doc_tree, id_calibrator)

        clause_idx = 0
        for clause_ids in doc_clause_ids:
            if len(clause_ids) > 1:
                # indices = [cid[-1] for cid in clause_ids[:-1]]

                indices = []
                for cid in clause_ids:
                    indices.append(cid[0] - 1)
                    indices.append(cid[-1])
                indices = set(indices)
                indices.remove(max(indices))
                indices.remove(min(indices))

                indices = sorted(list(indices))

                for word_idx in indices:
                    clause_idx = turn.search_clause_idx(word_idx, start_from=clause_idx)
                    turn.separate_clause(clause_idx, [word_idx])

        return turn

    def generate_dependency_tree(self, doc: Doc) -> List[DependencyTree]:
        doc_tree = []
        for sentence in doc.sents:
            dep_tree = DependencyTree.from_SpacySentence(sentence)
            doc_tree.append(dep_tree)

        return doc_tree

    def generate_id_calibrator(self, doc: Doc) -> List[List[int]]:
        id_calibrator = []
        idx = -1
        prev_offset = -1

        for sent in doc.sents: # 文ごとに処理
            calibrator = [-1] # calibrator を初期化

            if sent.text == ".":
                id_calibrator.append([-1, -1]) # もし文が "." だけの場合，id_calibrator に [-1, -1] を追加
                continue

            for word in sent: # 文内の単語ごとに処理
                start_char = word.idx # 単語の offset (str の開始・終了の index を取得)
                end_char = start_char + len(word)

                if start_char != prev_offset: # もし単語が subword に分割されていない場合，idx を更新
                    idx += 1

                calibrator.append(idx)

                prev_offset = end_char

                if word.dep_ == "punct": # punctuation が検出された場合
                    calibrator[-1] = -1 # calibrator の最後の idx を -1 へ設定

                    if word.i == sent[0].i: # もし，最初の1単語目が punct だった場合
                        idx -= 1 # idx を 1 減らす
                    continue

            id_calibrator.append(calibrator)

        return id_calibrator

    def separate_subordinate_clause(self, dep_tree: DependencyTree) -> List[DependencyTree]:
        id_queue = deque(dep_tree.are_dependencies_exist(self.__sc_marks))

        if len(id_queue) == 0:
            return [dep_tree]

        tree_queue = deque([dep_tree])
        sentence_trees = []

        word_id = id_queue.popleft()

        while len(tree_queue) > 0:
            tree = tree_queue.popleft()

            if word_id in tree.get_word_id():
                separated_tree = tree.separate(word_id)

                if len(id_queue) > 0:
                    tree_queue.append(tree)
                    tree_queue.append(separated_tree)
                    word_id = id_queue.popleft()
                else:
                    sentence_trees.append(tree)
                    sentence_trees.append(separated_tree)
                    word_id = None
            else:
                sentence_trees.append(tree)

        return sentence_trees

    def estimate_subordinate_clause_ids(
        self,
        doc_tree: List[DependencyTree],
        id_calibrator: List[List[int]]
    ) -> List[List[int]]:
        doc_clause_ids = []
        for dep_tree, calibrator in zip(doc_tree, id_calibrator):
            sentence_trees = self.separate_subordinate_clause(dep_tree)
            sentence_clause_ids = []

            for tree in sentence_trees:
                clause_ids = [calibrator[i] for i in tree.get_word_id()]
                if clause_ids[0] == -1:
                    clause_ids = clause_ids[1:]
                if clause_ids[-1] == -1:
                    clause_ids = clause_ids[:-1]

                # if len(clause_ids) == 0:
                #     continue

                sentence_clause_ids.append(clause_ids)

            doc_clause_ids.append(sentence_clause_ids)

        return doc_clause_ids

class StanzaClauseDetector(ModuleBase):
    def __init__(
        self,
        device: str ="cpu",
        subordinate_clause_marks: List[str] =SUBORDINATE_CLAUSE_MARKES
    ) -> 'StanzaClauseDetector':
        assert isinstance(subordinate_clause_marks, list), "subordinate_clause_marks must be list of str"

        super().__init__()
        self._model = StanzaDependencyParser(device=device)
        self.__sc_marks = subordinate_clause_marks

    def predict(self, turn: Turn) -> Turn:
        self._turn_type_checker(turn)

        turn.ignore_disfluency()
        text = ""
        for sentence in turn.clauses:
            text += (sentence.text + ". ")

        doc = self._model.predict(text)
        doc_tree = self.generate_dependency_tree(doc) # sentence ごとの dep tree のリスト
        id_calibrator = self.generate_id_calibrator(doc) # word id と doc id の補正
        doc_clause_ids = self.estimate_subordinate_clause_ids(doc_tree, id_calibrator)

        clause_idx = 0
        for clause_ids in doc_clause_ids:
            if len(clause_ids) > 1:
                # indices = [cid[-1] for cid in clause_ids[:-1]]

                indices = []
                for cid in clause_ids:
                    indices.append(cid[0] - 1)
                    indices.append(cid[-1])
                indices = set(indices)
                indices.remove(max(indices))
                indices.remove(min(indices))

                indices = sorted(list(indices))

                for word_idx in indices:
                    clause_idx = turn.search_clause_idx(word_idx, start_from=clause_idx)
                    turn.separate_clause(clause_idx, [word_idx])

        return turn

    def generate_dependency_tree(self, doc: Document) -> List[DependencyTree]:
        doc_tree = []
        for sentence in doc.sentences:
            dep_tree = DependencyTree.from_StanzaSentence(sentence)
            doc_tree.append(dep_tree)

        return doc_tree

    def generate_id_calibrator(self, doc: Document) -> List[List[int]]:
        id_calibrator = []
        idx = -1
        prev_offset = -1

        for sent in doc.sentences:
            calibrator = [-1]

            if sent.text == ".":
                id_calibrator.append([-1, -1])
                continue

            for word in sent.words:
                if word.start_char != prev_offset:
                    idx += 1

                calibrator.append(idx)

                prev_offset = word.end_char

                if word.deprel == "punct":
                    calibrator[-1] = -1
                    if word.id == 1:
                        idx -= 1
                    continue

            id_calibrator.append(calibrator)

        return id_calibrator

    def separate_subordinate_clause(self, dep_tree: DependencyTree) -> List[DependencyTree]:
        id_queue = deque(dep_tree.are_dependencies_exist(self.__sc_marks))

        if len(id_queue) == 0:
            return [dep_tree]

        tree_queue = deque([dep_tree])
        sentence_trees = []

        word_id = id_queue.popleft()

        while len(tree_queue) > 0:
            tree = tree_queue.popleft()

            if word_id in tree.get_word_id():
                separated_tree = tree.separate(word_id)

                if len(id_queue) > 0:
                    tree_queue.append(tree)
                    tree_queue.append(separated_tree)
                    word_id = id_queue.popleft()
                else:
                    sentence_trees.append(tree)
                    sentence_trees.append(separated_tree)
                    word_id = None
            else:
                sentence_trees.append(tree)

        return sentence_trees

    def estimate_subordinate_clause_ids(
        self,
        doc_tree: List[DependencyTree],
        id_calibrator: List[List[int]]
    ) -> List[List[int]]:
        doc_clause_ids = []
        for dep_tree, calibrator in zip(doc_tree, id_calibrator):
            sentence_trees = self.separate_subordinate_clause(dep_tree)
            sentence_clause_ids = []

            for tree in sentence_trees:
                clause_ids = [calibrator[i] for i in tree.get_word_id()]
                if clause_ids[0] == -1:
                    clause_ids = clause_ids[1:]
                if clause_ids[-1] == -1:
                    clause_ids = clause_ids[:-1]

                # if len(clause_ids) == 0:
                #     continue

                sentence_clause_ids.append(clause_ids)

            doc_clause_ids.append(sentence_clause_ids)

        return doc_clause_ids

