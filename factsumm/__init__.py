import logging
from itertools import permutations
from typing import Dict, List, Set, Tuple, Union

import pysbd
from factsumm.utils.level_entity import load_ner, load_rel
from factsumm.utils.level_sentence import load_qa, load_qg
from factsumm.utils.utils import Config
from rich import print

logging.getLogger("transformers").setLevel(logging.ERROR)


class FactSumm:

    def __init__(
        self,
        ner_model: str = None,
        rel_model: str = None,
        qg_model: str = None,
        qa_model: str = None,
    ):
        self.config = Config()
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

        # NER, RE, QG models supported by HuggingFace can be used (default can be found in `config.py`)
        ner = ner_model if ner_model is not None else self.config.NER_MODEL
        rel = rel_model if rel_model is not None else self.config.REL_MODEL
        qg = qg_model if qg_model is not None else self.config.QG_MODEL
        qa = qa_model if qa_model is not None else self.config.QA_MODEL

        # Load required pipes
        self.ner = load_ner(ner)
        self.rel = load_rel(rel)

        self.qg = load_qg(qg)
        self.qa = load_qa(qa)

    def build_comb(
        self,
        lines: List[str],
        total_entities: Union[List[Dict], List[List[Dict]]],
    ):
        total_combs = list()

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

    def _segment(self, text: str):
        return [line.strip() for line in self.segmenter.segment(text)]

    def _print_entities(self, mode: str, total_entities: List[List[Dict]]):
        print(f"{mode.upper()} Entities")
        for i, line_entities in enumerate(total_entities):
            print(
                f'{i+1}: {[(entity["word"], entity["entity"]) for entity in line_entities]}'
            )
        print()

    def _print_facts(self, mode: str, facts: Set[Tuple]):
        print(f"{mode.upper()} Facts")
        for fact in facts:
            print(fact)
        print()

    def _print_qas(self, mode: str, questions: List[Dict]):
        print(f"{mode.upper()} Questions")
        for question in questions:
            print(
                f"[Q] {question['question']}\t[A] {question['answer']}\t[Pred] {question['prediction']['answer']}"
            )
        print()

    def __call__(self, source: str, summary: str):
        source_lines = self._segment(source)
        summary_lines = self._segment(summary)

        # extract per-line entities
        source_ents = self.ner(source_lines)
        summary_ents = self.ner(summary_lines)

        self._print_entities("source", source_ents)
        self._print_entities("summary", summary_ents)

        # extract entity-based triple: (head, relation, tail)
        source_facts = self.count_facts(source_lines, source_ents)
        summary_facts = self.count_facts(summary_lines, summary_ents)

        common_facts = summary_facts.intersection(source_facts)
        diff_facts = summary_facts.difference(source_facts)

        self._print_facts("source", source_facts)
        self._print_facts("summary", summary_facts)

        self._print_facts("common", common_facts)
        self._print_facts("diff", diff_facts)

        source_qas = self.qg(source_lines, source_ents)
        summary_qas = self.qg(summary_lines, summary_ents)

        source_answers = self.qa(source, source_qas)
        summary_answers = self.qa(summary, summary_qas)

        self._print_qas("source", source_answers)
        self._print_qas("summary", summary_answers)
