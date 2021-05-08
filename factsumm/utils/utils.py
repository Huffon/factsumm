from dataclasses import dataclass

from transformers import pipeline


@dataclass
class Config:
    NER_MODEL: str = "elastic/distilbert-base-cased-finetuned-conll03-english"
    REL_MODEL: str = "studio-ousia/luke-large-finetuned-tacred"
    QG_MODEL: str = "mrm8488/t5-base-finetuned-question-generation-ap"
    QA_MODEL: str = "distilbert-base-uncased-distilled-squad"
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
