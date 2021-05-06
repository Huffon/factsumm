from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple

from transformers import pipeline


@dataclass
class Config:
    NER_MODEL: str = "elastic/distilbert-base-cased-finetuned-conll03-english"
    REL_MODEL: str = "studio-ousia/luke-large-finetuned-tacred"
    QG_MODEL: str = "iarfmoose/t5-base-question-generator"
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
