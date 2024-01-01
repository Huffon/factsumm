import re
import string
from collections import Counter
from typing import List

from transformers import pipeline


class Config:
    NER_MODEL: str = "tner/deberta-v3-large-ontonotes5"
    REL_MODEL: str = "studio-ousia/luke-large-finetuned-tacred"
    QG_MODEL: str = "mrm8488/t5-base-finetuned-question-generation-ap"
    QA_MODEL: str = "deepset/roberta-base-squad2"
    SUMM_MODEL: str = "sshleifer/distilbart-cnn-12-6"


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


def f1_score(gold_answer: str, pred_answer: str) -> float:
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
        return 0.0

    precision = 1.0 * num_same_toks / len(pred_toks)
    recall = 1.0 * num_same_toks / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def score_qags(source_answers: List, summary_answers: List) -> float:
    """
    Caculate QAGS Score

        See also https://arxiv.org/abs/2004.04228

    Args:
        source_answers (List): source answers selected based on source document
        summary_answers (List): summary answers selected based on generated summary

    """
    scores = []

    for source_answer, summary_answer in zip(source_answers, summary_answers):
        source_answer = source_answer["prediction"]
        summary_answer = summary_answer["prediction"]
        scores.append(f1_score(source_answer, summary_answer))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)
