import os
import logging
from itertools import permutations
from typing import Dict, List, Optional, Set, Tuple, Union

import pysbd
from sumeval.metrics.rouge import RougeCalculator

from factsumm.utils.module_entity import load_ner, load_rel
from factsumm.utils.module_question import load_qa, load_qg
from factsumm.utils.module_sentence import load_bert_score
from factsumm.utils.utils import Config, qags_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)


class FactSumm:

    def __init__(
        self,
        ner_model: Optional[str] = None,
        rel_model: Optional[str] = None,
        qg_model: Optional[str] = None,
        qa_model: Optional[str] = None,
        bert_score_model: Optional[str] = None,
    ):
        """
        FactSumm object used to calculate Factual Consistency score of Abstractive Summarization model

        Args:
            ner_model (str, optional): Named Entity Recognition model to be used (HuggingFace). Defaults to None.
            rel_model (str, optional): Relation Extraction model to be used (HuggingFace). Defaults to None.
            qg_model (str, optional): Question Answering model to be used (HuggingFace). Defaults to None.
            qa_model (str, optional): Question Genration model to be used (HuggingFace). Defaults to None.
            bert_score_model (str, optional): BERTScore model to be used (HuggingFace). Defaults to None.

        """
        self.config = Config()
        self.sentence_segmenter = pysbd.Segmenter(language="en", clean=False)
        self.rouge = RougeCalculator(stopwords=True, lang="en")

        # NER, RE, QG, QA models supported by HuggingFace can be used (default models can be found in `config.py`)
        self.ner = ner_model if ner_model is not None else self.config.NER_MODEL
        self.rel = rel_model if rel_model is not None else self.config.REL_MODEL
        self.qg = qg_model if qg_model is not None else self.config.QG_MODEL
        self.qa = qa_model if qa_model is not None else self.config.QA_MODEL
        self.bert_score = bert_score_model

    def build_perm(
        self,
        lines: List[str],
        total_entities: Union[List[Dict], List[List[Dict]]],
    ) -> List:
        """
        Build entity permutations for Relation Extraction

        Args:
            lines (List[str]): segmented document lines
            total_entities (Union[List[Dict], List[List[Dict]]]): list of total entities

        Returns:
            List: list of permutations

        """
        total_perms = []

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

    def get_facts(self, lines: List[str], entities: List[List[Dict]]) -> Set:
        """
        Get fact triples using Relation Extraction model

        Args:
            lines (List[str]): segmented document lines
            entities (List[List[Dict]]): list of total entities

        Returns:
            Set: set of relation inferenced from permutations

        """
        perms = self.build_perm(lines, entities)
        triples = []

        for perm, entity in zip(perms, entities):
            entity_key = {ent["word"].replace("▁", ""): ent["entity"] for ent in entity}
            facts = self.rel(perm)
            filtered_facts = []

            for fact in facts:
                head, relation, tail = fact

                head = head.strip()
                tail = tail.strip()

                head_entity_type = entity_key.get(head, None)
                tail_entity_type = entity_key.get(tail, None)

                if head_entity_type is not None and head_entity_type == "PERSON" and not relation.startswith("per:"):
                    continue

                if head_entity_type is not None and head_entity_type != "PERSON" and relation.startswith("per:"):
                    continue

                if tail_entity_type is not None and tail_entity_type != "PERSON" and "members" in relation:
                    continue

                filtered_facts.append(tuple([head, relation, tail]))

            triples.extend(filtered_facts)

        return set(triples)

    def _segment_sentence(self, text: str) -> List[str]:
        """
        Segment input text into (possibly) multiple sentences

        Args:
            text (str): text to be segmented

        Returns:
            List[str]: list of segmented lines

        """
        return [line.strip() for line in self.sentence_segmenter.segment(text)]

    def _print_entities(self, mode: str, total_entities: List[List[Dict]]):
        logging.info("<%s Entities>", mode.capitalize())
        for i, line_entities in enumerate(total_entities):
            logging.info("Line No.%s: [%s]", i+1, [(entity["word"].replace("▁", ""), entity["entity"]) for entity in line_entities])
        logging.info("")

    def calculate_rouge(
        self,
        source: str,
        summary: str,
    ) -> Tuple[float, float, float]:
        """
        Calculate ROUGE score

        Args:
            source (str): original source
            summary (str): generated summary

        Returns:
            Tuple: (ROUGE-1, ROUGE-2, ROUGE-L) tuple

        """
        source_lines = self._segment_sentence(source)

        rouge_1 = self.rouge.rouge_n(summary, source_lines, 1)
        rouge_2 = self.rouge.rouge_n(summary, source_lines, 2)
        rouge_l = self.rouge.rouge_l(summary, source_lines)

        logging.info("Avg. ROUGE-1: %s\nAvg. ROUGE-2: %s\nAvg. ROUGE-L: %s", rouge_1, rouge_2, rouge_l)
        return rouge_1, rouge_2, rouge_l

    def _print_facts(self, mode: str, facts: Set[Tuple]):
        logging.info("<%s Facts>", mode.capitalize())
        for fact in facts:
            logging.info(fact)
        logging.info("")

    def _filter_out(self, sources: Set, summaries: Set) -> Tuple[Set, Set]:
        """
        Filter out triples that don't share a subject and relation for comparability

        Args:
            sources (Set): set of triples from source
            summaries (Set): set of triples from summary

        Returns:
            Tuple[Set, Set]: filtered sources and summaries

        """
        source_tuple = {(source[0], source[1]) for source in sources}
        summary_tuple = {(summary[0], summary[1]) for summary in summaries}

        sources = {
            source for source in sources
            if (source[0], source[1]) in summary_tuple
        }
        summaries = {
            summary for summary in summaries
            if (summary[0], summary[1]) in source_tuple
        }
        return sources, summaries

    def extract_facts(
        self,
        source: str,
        summary: str,
        verbose: bool = False,
        device: str = "cpu",
    ):
        """
        Extract (head_entity, relation, tail_entity) relation triple using NER & RE module

            See also https://arxiv.org/abs/1905.13322.pdf

        Args:
            source (str): original source
            summary (str): generated summary
            verbose (bool, optional): print verbose option. Defaults to False.
            device (str): device info

        """
        if isinstance(self.ner, str):
            self.ner = load_ner(self.ner, device)

        if isinstance(self.rel, str):
            self.rel = load_rel(self.rel, device)

        source_lines = self._segment_sentence(source)
        summary_lines = self._segment_sentence(summary)

        # extract per-line named entities
        source_entities = self.ner(source_lines)
        summary_entities = self.ner(summary_lines)

        # extract entity-based triple: (head, relation, tail)
        source_facts = self.get_facts(source_lines, source_entities)
        summary_facts = self.get_facts(summary_lines, summary_entities)

        # filter out some facts
        source_facts, summary_facts = self._filter_out(
            source_facts,
            summary_facts,
        )

        common_facts = summary_facts.intersection(source_facts)
        diff_facts = summary_facts.difference(source_facts)

        if verbose:
            self._print_entities("source", source_entities)
            self._print_entities("summary", summary_entities)

            self._print_facts("source", source_facts)
            self._print_facts("summary", summary_facts)

            self._print_facts("common", common_facts)
            self._print_facts("diff", diff_facts)

        if not summary_facts:
            fact_score = 0.0
        else:
            fact_score = len(common_facts) / len(summary_facts)
        logging.info("Fact Score: %s", fact_score)

        return source_entities, summary_entities, fact_score

    def _print_qas(self, mode: str, questions: List[Dict]):
        logging.info("Answers based on %s (Questions are generated from Summary)", mode.capitalize())
        for question in questions:
            logging.info("[Q] %s\t[Pred] %s", question["question"], question["prediction"])
        logging.info("")

    def extract_qas(
        self,
        source: str,
        summary: str,
        source_ents: List = None,
        summary_ents: List = None,
        verbose: bool = False,
        device: str = "cpu",
    ) -> float:
        """
        Extract Question & Answering Pair generated from Question Generation module

            See also https://arxiv.org/abs/2004.04228

        Args:
            source (str): original source
            summary (str): generated summary
            source_ents (List, optional): named entities extracted from source. Defaults to None.
            summary_ents (List, optional): named entities extracted from source. Defaults to None.
            verbose (bool, optional): print verbose option. Defaults to False.
            device (str): device info

        """
        if isinstance(self.qg, str) and isinstance(self.qa, str):
            self.qg = load_qg(self.qg, device)
            self.qa = load_qa(self.qa, device)

        if isinstance(self.ner, str):
            self.ner = load_ner(self.ner, device)

        source_lines = self._segment_sentence(source)
        summary_lines = self._segment_sentence(summary)

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
        logging.info("QAGS Score: %s\n", qa_score)

        return qa_score

    def calculate_bert_score(
        self,
        source: str,
        summary: str,
        device: str = "cpu",
    ) -> Dict[str, float]:
        """
        Calculate BERTScore

            See also https://arxiv.org/abs/2005.03754

        Args:
            source (str): original source
            summary (str): generated summary
            device (str): device info

        Returns:
            Dict: (Precision, Recall, F1) BERTScore dictionary

        """
        if self.bert_score is None:
            self.bert_score = load_bert_score(device)

        source_lines = self._segment_sentence(source)
        summary_lines = self._segment_sentence(summary)

        scores = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

        for summary_line in summary_lines:
            precision, recall, f1 = self.bert_score([summary_line], [source_lines])
            scores["precision"] += precision.item()
            scores["recall"] += recall.item()
            scores["f1"] += f1.item()

        if len(summary_lines) > 1:
            scores["precision"] /= len(summary_lines)
            scores["recall"] /= len(summary_lines)
            scores["f1"] /= len(summary_lines)

        logging.info("<BERTScore Score>\nPrecision: %s\nRecall: %s\nF1: %s", scores["precision"], scores["recall"], scores["f1"])

        return scores

    def __call__(
        self,
        sources: Union[List[str], str],
        summaries: Union[List[str], str],
        verbose: bool = False,
        device: str = "cpu",
    ) -> Dict:
        if isinstance(sources, str) and isinstance(summaries, str):
            sources = [sources]
            summaries = [summaries]

        if len(sources) != len(summaries):
            raise ValueError("`sources` and `summaries` must have the same number of elements!")

        num_pairs = len(sources)

        fact_scores = 0
        qags_scores = 0
        rouge_scores = {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
        }
        bert_scores = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

        for source, summary in zip(sources, summaries):
            source_ents, summary_ents, fact_score = self.extract_facts(
                source,
                summary,
                verbose,
                device,
            )
            fact_scores += fact_score

            qags_score = self.extract_qas(
                source,
                summary,
                source_ents,
                summary_ents,
                verbose,
                device,
            )
            qags_scores += qags_score

            rouge_1, rouge_2, rouge_l = self.calculate_rouge(source, summary)
            rouge_scores["rouge-1"] += rouge_1
            rouge_scores["rouge-2"] += rouge_2
            rouge_scores["rouge-l"] += rouge_l

            bert_scores = self.calculate_bert_score(source, summary, device)

        return {
            "fact_score": fact_scores / num_pairs,
            "qa_score": qags_scores / num_pairs,
            "rouge": rouge_scores,
            "bert_score": bert_scores,
        }
