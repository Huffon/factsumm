from dataclasses import dataclass
from typing import Dict, List

from transformers import pipeline


@dataclass
class Config:
    NER_MODEL: str = "elastic/distilbert-base-cased-finetuned-conll03-english"
    REL_MODEL: str = "studio-ousia/luke-large-finetuned-tacred"
    QG_MODEL: str = "mrm8488/t5-base-finetuned-question-generation-ap"
    QA_MODEL: str = "deepset/roberta-base-squad2"
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
