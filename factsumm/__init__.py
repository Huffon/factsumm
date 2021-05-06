from itertools import permutations
from typing import Dict, List, Union

import pysbd
from rich import print

from factsumm.utils.level_entity import load_ner, load_rel
from factsumm.utils.level_sentence import load_qg
from factsumm.utils.utils import Config


class FactSumm:

    def __init__(
        self,
        ner_model: str = None,
        rel_model: str = None,
        qg_model: str = None,
    ):
        self.config = Config()
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

        # NER, RE, QG models supported by HuggingFace can be used (default can be found in `config.py`)
        ner = ner_model if ner_model is not None else self.config.NER_MODEL
        rel = rel_model if rel_model is not None else self.config.REL_MODEL
        qg = qg_model if qg_model is not None else self.config.QG_MODEL

        # Load required pipes
        self.ner = load_ner(ner)
        self.rel = load_rel(rel)
        self.qg = load_qg(qg)

    def build_comb(
        self,
        lines: List[str],
        total_entities: Union[List[Dict], List[List[Dict]]],
    ):
        total_combs = list()

        if isinstance(total_entities[0], dict):
            total_entities = [total_entities]

        for line, line_entities in zip(lines, total_entities):
            line_combs = list(permutations(line_entities, 2))

            line_combs = [{
                "text":
                    line,
                "spans": [
                    (comb[0]["start"], comb[0]["end"]),
                    (comb[-1]["start"], comb[-1]["end"]),
                ]
            } for comb in line_combs]

            total_combs.append(line_combs)

        return total_combs

    def count_facts(self, lines: List[str], entities: List[List[Dict]]):
        combs = self.build_comb(lines, entities)

        triples = list()

        for comb in combs:
            triples.extend(self.rel(comb))

        return set(triples)

    def count_question(self):
        pass

    def __call__(self, source: str, summary: str):
        source_lines = self.segmenter.segment(source)
        summary_lines = self.segmenter.segment(summary)

        # extract per-line entities
        source_ents = self.ner(source_lines)
        summary_ents = self.ner(summary_lines)

        print(f"[DOC] Document Entities {source_ents}")
        print(f"[SUM] Summary Entities {summary_ents}\n")

        # extract entity-based triple: (head, relation, tail)
        source_facts = self.count_facts(source_lines, source_ents)
        summary_facts = self.count_facts(summary_lines, summary_ents)

        common_facts = summary_facts.intersection(source_facts)
        diff_facts = summary_facts.difference(source_facts)

        print(f"[DOC] Document Facts {source_facts}")
        print(f"[SUM] Summary Facts {summary_facts}\n")

        print(f"Common Facts {common_facts}")
        print(f"Diff Facts {diff_facts}\n")

        # source_questions = self.qg(source)
        # summary_questions = self.qg(summary)


if __name__ == "__main__":
    scorer = FactSumm()
    scorer(
        "Inception is a science fiction film directed by Nolan and starring Leonardo.",
        "Leonardo directed the action film Inception.",
    )
