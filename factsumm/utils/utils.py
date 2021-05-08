from dataclasses import dataclass
from typing import Dict, List

from transformers import pipeline


@dataclass
class Config:
    NER_MODEL: str = "elastic/distilbert-base-cased-finetuned-conll03-english"
    REL_MODEL: str = "studio-ousia/luke-large-finetuned-tacred"
    QG_MODEL: str = "mrm8488/t5-base-finetuned-question-generation-ap"
    QA_MODEL: str = "distilbert-base-uncased-distilled-squad"
    SUMM_MODEL: str = "sshleifer/distilbart-cnn-12-6"


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

    def _append(lst: List, word: str, type: str):
        if prev_word != "":
            lst.append((word, type))

    result = list()

    prev_word = entities[0]["word"]

    prev_entity = entities[0]["entity"]
    prev_type = _remove_prefix(prev_entity)

    for pair in entities[1:]:
        word = pair["word"]
        entity = pair["entity"]
        type = _remove_prefix(entity)

        if "##" in word:
            prev_word += word
            continue

        if entity == prev_entity:
            if entity == "O":
                _append(result, prev_word, prev_type)
                result.append((word, type))
                prev_word = ""
            if "I-" in entity:
                prev_word += f" {word}"
        elif (entity != prev_entity) and ("I-" in entity) and (type != "O"):
            prev_word += f" {word}"
        else:
            _append(result, prev_word, prev_type)
            prev_word = word
            prev_type = type

        prev_entity = entity

    _append(result, prev_word, prev_type)

    return [(pair[0].replace("##", ""), pair[1])
            for pair in result
            if pair[1] != "O"]


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
