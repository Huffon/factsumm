import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from rich import print
from transformers import pipeline


@dataclass
class Config:
    NER_MODEL: str = "flair/ner-english-ontonotes-fast"
    REL_MODEL: str = "studio-ousia/luke-large-finetuned-tacred"
    QG_MODEL: str = "mrm8488/t5-base-finetuned-question-generation-ap"
    QA_MODEL: str = "deepset/roberta-base-squad2"
    SUMM_MODEL: str = "sshleifer/distilbart-cnn-12-6"
    BERT_SCORE_MODEL: str = "microsoft/deberta-base-mnli"


def grouped_entities(entities: List[Dict]):
    """
    Group entities to concatenate BIO

    Args:
        entities (List[Dict]): list of inference entities

    Returns:
        List[Tuple]: list of grouped BIO scheme entities

    """

    def _remove_prefix(entity: str) -> str:
        if "-" in entity:
            entity = entity[2:]
        return entity

    def _append(lst: List, word: str, type: str, start: int, end: int):
        if prev_word != "":
            lst.append((word, type, start, end))

    result = list()

    prev_word = entities[0]["word"]
    prev_entity = entities[0]["entity"]
    prev_type = _remove_prefix(prev_entity)
    prev_start = entities[0]["start"]
    prev_end = entities[0]["end"]

    for pair in entities[1:]:
        word = pair["word"]
        entity = pair["entity"]
        type = _remove_prefix(entity)
        start = pair["start"]
        end = pair["end"]

        if "##" in word:
            prev_word += word
            prev_end = end
            continue

        if entity == prev_entity:
            if entity == "O":
                _append(result, prev_word, prev_type, prev_start, prev_end)
                result.append((word, type))
                prev_word = ""
                prev_start = start
                prev_end = end
            if "I-" in entity:
                prev_word += f" {word}"
                prev_end = end
        elif (entity != prev_entity) and ("I-" in entity) and (type != "O"):
            prev_word += f" {word}"
            prev_end = end
        else:
            _append(result, prev_word, prev_type, prev_start, prev_end)
            prev_word = word
            prev_type = type
            prev_start = start
            prev_end = end

        prev_entity = entity

    _append(result, prev_word, prev_type, prev_start, prev_end)

    cache = dict()
    dedup = list()

    for pair in result:
        if pair[1] == "O":
            continue

        if pair[0] not in cache:
            dedup.append({
                "word": pair[0].replace("##", ""),
                "entity": pair[1],
                "start": pair[2],
                "end": pair[3]
            })
            cache[pair[0]] = None
    return dedup


def load_summarizer(model: str) -> object:
    """
    Load Summarization model from HuggingFace hub

    Args:
        model (str): model name to be loaded

    Returns:
        object: Pipeline-based Summarization model

    """
    return pipeline(
        "summarization",
        model=model,
        tokenizer=model,
        framework="pt",
    )


def f1_score(gold_answer: str, pred_answer: str):
    """
    Calculate token-level F1 score

        See also https://github.com/W4ngatang/qags/blob/master/qa_utils.py#L43

    Args:
        gold_answer (str): answer selected based on source document
        pred_answer (str): answer selected based on generated summary

    """

    def _normalize_answer(text: str):

        def _remove_punc(text: str):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def _remove_articles(text: str):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def _white_space_fix(text: str):
            return " ".join(text.split())

        return _white_space_fix(_remove_articles(_remove_punc(text.lower())))

    gold_toks = _normalize_answer(gold_answer).split()
    pred_toks = _normalize_answer(pred_answer).split()

    common_toks = Counter(gold_toks) & Counter(pred_toks)

    num_same_toks = sum(common_toks.values())

    # If either is <unanswerable>, then F1 is 1 if they agree, 0 otherwise
    if gold_answer == "<unanswerable>" or pred_answer == "<unanswerable>":
        return int(gold_answer == pred_answer)

    if num_same_toks == 0:
        return 0

    precision = 1.0 * num_same_toks / len(pred_toks)
    recall = 1.0 * num_same_toks / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qags_score(source_answers: List, summary_answers: List):
    """
    Caculate QAGS Score

        See also https://arxiv.org/abs/2004.04228

    Args:
        source_answers (List): source answers selected based on source document
        summary_answers (List): summary answers selected based on generated summary

    """
    scores = list()

    for source_answer, summary_answer in zip(source_answers, summary_answers):
        source_answer = source_answer["prediction"]
        summary_answer = summary_answer["prediction"]
        scores.append(f1_score(source_answer, summary_answer))

    return sum(scores) / len(scores)


if __name__ == "__main__":
    model = "elastic/distilbert-base-cased-finetuned-conll03-english"

    article = "Marie Curie was a Polish and French physicist and chemist who conducted pioneering research on radioactivity. As the first of the Curie family legacy of five Nobel Prizes, she was the first woman to win a Nobel Prize, the first person and the only woman to win the Nobel Prize twice, and the only person to win the Nobel Prize in two scientific fields. She was the first woman to become a professor at the University of Paris in 1906. She was born in Warsaw, in what was then the Kingdom of Poland, part of the Russian Empire. She studied at Warsaw's clandestine Flying University and began her practical scientific training in Warsaw. In 1891, aged 24, she followed her elder sister Bronisława to study in Paris, where she earned her higher degrees and conducted her subsequent scientific work. In 1895 she married the French physicist Pierre Curie, and she shared the 1903 Nobel Prize in Physics with him and with the physicist Henri Becquerel for their pioneering work developing the theory of 'radioactivity'—a term she coined. In 1906 Pierre Curie died in a Paris street accident. Marie won the 1911 Nobel Prize in Chemistry for her discovery of the elements polonium and radium, using techniques she invented for isolating radioactive isotopes."

    ner = pipeline(
        task="ner",
        model=model,
        tokenizer=model,
        ignore_labels=[],
        framework="pt",
    )

    tags = ner(article)
    print(grouped_entities(tags))
