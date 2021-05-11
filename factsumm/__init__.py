import logging
import os
from itertools import permutations
from typing import Dict, List, Set, Tuple, Union

import pysbd
from rich import print
from sumeval.metrics.rouge import RougeCalculator

from factsumm.utils.level_entity import load_ie, load_ner, load_rel
from factsumm.utils.level_sentence import load_qa, load_qg
from factsumm.utils.utils import Config

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
        self.rouge = RougeCalculator(stopwords=True, lang="en")

        # NER, RE, QG models supported by HuggingFace can be used (default can be found in `config.py`)
        self.ner = ner_model if ner_model is not None else self.config.NER_MODEL
        self.rel = rel_model if rel_model is not None else self.config.REL_MODEL
        self.qg = qg_model if qg_model is not None else self.config.QG_MODEL
        self.qa = qa_model if qa_model is not None else self.config.QA_MODEL
        self.ie = None

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

    def _segment(self, text: str):
        return [line.strip() for line in self.segmenter.segment(text)]

    def _print_entities(self, mode: str, total_entities: List[List[Dict]]):
        print(f"{mode.upper()} Entities")
        for i, line_entities in enumerate(total_entities):
            print(
                f'{i+1}: {[(entity["word"], entity["entity"]) for entity in line_entities]}'
            )
        print()

    def calculate_rouge(self, source: str, summary: str):
        rouge_1 = self.rouge.rouge_n(source, summary, 1)
        rouge_2 = self.rouge.rouge_n(source, summary, 2)
        rouge_l = self.rouge.rouge_l(source, summary)
        return rouge_1, rouge_2, rouge_l

    def _print_facts(self, mode: str, facts: Set[Tuple]):
        print(f"{mode.upper()} Facts")
        for fact in facts:
            print(fact)
        print()

    def extract_facts(self, source: str, summary: str, verbose: bool = False):
        if isinstance(self.ner, str) and isinstance(self.rel, str):
            self.ner = load_ner(self.ner)
            self.rel = load_rel(self.rel)

        source_lines = self._segment(source)
        summary_lines = self._segment(summary)

        # extract per-line entities
        source_ents = self.ner(source_lines)
        summary_ents = self.ner(summary_lines)

        # extract entity-based triple: (head, relation, tail)
        source_facts = self.count_facts(source_lines, source_ents)
        summary_facts = self.count_facts(summary_lines, summary_ents)

        common_facts = summary_facts.intersection(source_facts)
        diff_facts = summary_facts.difference(source_facts)

        if verbose:
            self._print_entities("source", source_ents)
            self._print_entities("summary", summary_ents)

            self._print_facts("source", source_facts)
            self._print_facts("summary", summary_facts)

            self._print_facts("common", common_facts)
            self._print_facts("diff", diff_facts)

        fact_score = len(common_facts) / len(summary_facts)
        print(f"Fact Score: {fact_score}")

        return source_ents, summary_ents, fact_score

    def _print_qas(self, mode: str, questions: List[Dict]):
        print(f"{mode.upper()} Questions")
        for question in questions:
            print(
                f"[Q] {question['question']}\t[A] {question['answer']}\t[Pred] {question['prediction']}"
            )
        print()

    def extract_qas(
        self,
        source: str,
        summary: str,
        source_ents: List = None,
        summary_ents: List = None,
        verbose: bool = False,
    ):
        if isinstance(self.qg, str) and isinstance(self.qa, str):
            self.qg = load_qg(self.qg)
            self.qa = load_qa(self.qa)

        if isinstance(self.ner, str):
            self.ner = load_ner(self.ner)

        source_lines = self._segment(source)
        summary_lines = self._segment(summary)

        if source_ents is None:
            source_ents = self.ner(source_lines)

        if summary_ents is None:
            summary_ents = self.ner(summary_lines)

        source_qas = self.qg(source_lines, source_ents)
        summary_qas = self.qg(summary_lines, summary_ents)

        source_answers = self.qa(source, source_qas)
        summary_answers = self.qa(summary, summary_qas)
        diff_answers = self.qa(summary, source_qas)

        if verbose:
            self._print_qas("source", source_answers)
            self._print_qas("summary", summary_answers)
            self._print_qas("diff", diff_answers)

    def _print_triples(self, mode: str, triples: Set):
        print(f"{mode.upper()} Triples")
        for triple in triples:
            print(triple)
        print()

    def extract_triples(self, source: str, summary: str, verbose: bool = False):
        if self.ie is None:
            self.ie = load_ie()

        source_triples = {(
            triple["subject"],
            triple["relation"],
            triple["object"],
        ) for triple in self.ie(source)}

        summary_triples = {(
            triple["subject"],
            triple["relation"],
            triple["object"],
        ) for triple in self.ie(summary)}

        if verbose:
            self._print_triples("source", source_triples)
            self._print_triples("summary", summary_triples)

        common_triples = summary_triples.intersection(source_triples)
        triple_score = len(common_triples) / len(summary_triples)

        print(f"Triple Score: {triple_score}")

        return triple_score

    def __call__(self, source: str, summary: str, verbose: bool = False):
        source_ents, summary_ents, fact_score = self.extract_facts(
            source,
            summary,
            verbose,
        )
        self.extract_qas(source, summary, source_ents, summary_ents, verbose)
        triple_score = self.extract_triples(source, summary, verbose)
