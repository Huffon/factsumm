import logging
import os
from itertools import permutations
from typing import Dict, List, Set, Tuple, Union

import pysbd
from rich import print
from sumeval.metrics.rouge import RougeCalculator

from factsumm.utils.level_entity import load_ie, load_ner, load_rel
from factsumm.utils.level_sentence import load_qa, load_qg
from factsumm.utils.utils import Config, qags_score

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

        # NER, RE, QG, QA models supported by HuggingFace can be used (default can be found in `config.py`)
        self.ner = ner_model if ner_model is not None else self.config.NER_MODEL
        self.rel = rel_model if rel_model is not None else self.config.REL_MODEL
        self.qg = qg_model if qg_model is not None else self.config.QG_MODEL
        self.qa = qa_model if qa_model is not None else self.config.QA_MODEL
        self.ie = None

    def build_perm(
        self,
        lines: List[str],
        total_entities: Union[List[Dict], List[List[Dict]]],
    ):
        """
        Build entity permutations for Relation Extraction

        Args:
            lines (List[str]): segmented document lines
            total_entities (Union[List[Dict], List[List[Dict]]]): list of total entities

        Returns:
            List: list of permutations

        """
        total_perms = list()

        for line, line_entities in zip(lines, total_entities):
            line_perms = list(permutations(line_entities, 2))

            line_perms = [{
                "text":
                    line,
                "spans": [
                    (comb[0]["start"], comb[0]["end"]),
                    (comb[-1]["start"], comb[-1]["end"]),
                ]
            } for comb in line_perms]

            total_perms.append(line_perms)

        return total_perms

    def get_facts(self, lines: List[str], entities: List[List[Dict]]):
        """
        Get fact triples using Relation Extraction model

        Args:
            lines (List[str]): segmented document lines
            entities (List[List[Dict]]): list of total entities

        Returns:
            Set: set of relation inferenced from permutations

        """
        perms = self.build_perm(lines, entities)
        triples = list()

        for perm in perms:
            triples.extend(self.rel(perm))

        return set(triples)

    def _segment(self, text: str):
        return [line.strip() for line in self.segmenter.segment(text)]

    def _print_entities(self, mode: str, total_entities: List[List[Dict]]):
        # yapf:disable
        print(f"{mode.upper()} Entities")
        for i, line_entities in enumerate(total_entities):
            print(f'{i+1}: {[(entity["word"], entity["entity"]) for entity in line_entities]}')
        print()
        # yapf:enable

    def calculate_rouge(self, source: str, summary: str):
        """
        Calculate ROUGE score

        Args:
            source (str): original source
            summary (str): generated summary

        Returns:
            Tuple: (ROUGE-1, ROUGE-2, ROUGE-L) tuple

        """
        source_lines = self._segment(source)
        num_lines = len(source_lines)

        rouge_1 = 0.0
        rouge_2 = 0.0
        rouge_l = 0.0
        for source_line in source_lines:
            rouge_1 += self.rouge.rouge_n(summary, source_line, 1)
            rouge_2 += self.rouge.rouge_n(summary, source_line, 2)
            rouge_l += self.rouge.rouge_l(summary, source_line)

        avg_rouge_1 = rouge_1 / num_lines
        avg_rouge_2 = rouge_2 / num_lines
        avg_rouge_l = rouge_l / num_lines

        print(
            f"Avg. ROUGE-1: {avg_rouge_1}\nAvg. ROUGE-2: {avg_rouge_2}\nAvg. ROUGE-L: {avg_rouge_l}"
        )
        return avg_rouge_1, avg_rouge_2, avg_rouge_l

    def _print_facts(self, mode: str, facts: Set[Tuple]):
        print(f"{mode.upper()} Facts")
        for fact in facts:
            print(fact)
        print()

    def extract_facts(self, source: str, summary: str, verbose: bool = False):
        """
        Extract (head_entity, relation, tail_entity) relation triple using NER & RE module

            See also https://arxiv.org/abs/1905.13322.pdf

        Args:
            source (str): original source
            summary (str): generated summary
            verbose (bool, optional): print verbose option. Defaults to False.

        """
        if isinstance(self.ner, str) and isinstance(self.rel, str):
            self.ner = load_ner(self.ner)
            self.rel = load_rel(self.rel)

        source_lines = self._segment(source)
        summary_lines = self._segment(summary)

        # extract per-line entities
        source_ents = self.ner(source_lines)
        summary_ents = self.ner(summary_lines)

        # extract entity-based triple: (head, relation, tail)
        source_facts = self.get_facts(source_lines, source_ents)
        summary_facts = self.get_facts(summary_lines, summary_ents)

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
        # yapf:disable
        print(f"Answers based on {mode.upper()} (Questions are generated from Summary)")
        for question in questions:
            print(f"[Q] {question['question']}\t[Pred] {question['prediction']}")
        print()
        # yapf:enable

    def extract_qas(
        self,
        source: str,
        summary: str,
        source_ents: List = None,
        summary_ents: List = None,
        verbose: bool = False,
    ):
        """
        Extract Question & Answering Pair generated from Question Generation module

            See also https://arxiv.org/abs/2004.04228

        Args:
            source (str): original source
            summary (str): generated summary
            source_ents (List, optional): named entities extracted from source. Defaults to None.
            summary_ents (List, optional): named entities extracted from source. Defaults to None.
            verbose (bool, optional): print verbose option. Defaults to False.

        """
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

        summary_qas = self.qg(summary_lines, summary_ents)

        source_answers = self.qa(source, summary_qas)
        summary_answers = self.qa(summary, summary_qas)

        if verbose:
            self._print_qas("source", source_answers)
            self._print_qas("summary", summary_answers)

        qa_score = qags_score(source_answers, summary_answers)
        print(f"QAGS Score: {qa_score}\n")

        return qa_score

    def _print_triples(self, mode: str, triples: Set):
        print(f"{mode.upper()} Triples")
        for triple in triples:
            print(triple)
        print()

    def extract_triples(self, source: str, summary: str, verbose: bool = False):
        """
        Extract OpenIE based fact triples

        Args:
            source (str): original source
            summary (str): generated summary
            verbose (bool, optional): print verbose option. Defaults to False.

        """
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

        print(f"Triple Score: {triple_score}\n")

        return triple_score

    def __call__(self, source: str, summary: str, verbose: bool = False):
        source_ents, summary_ents, fact_score = self.extract_facts(
            source,
            summary,
            verbose,
        )
        qags_score = self.extract_qas(
            source,
            summary,
            source_ents,
            summary_ents,
            verbose,
        )
        triple_score = self.extract_triples(source, summary, verbose)
        rouge_1, rouge_2, rouge_l = self.calculate_rouge(source, summary)

        return {
            "fact_score": fact_score,
            "qa_score": qags_score,
            "triple_score": triple_score,
            "rouge": (rouge_1, rouge_2, rouge_l),
        }
