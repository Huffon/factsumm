import logging
from typing import List, Tuple

from requests import HTTPError
from transformers import LukeForEntityPairClassification, LukeTokenizer, pipeline

from factsumm.utils.utils import grouped_entities


def load_ner(model: str, device: str) -> object:
    """
    Load Named Entity Recognition model from HuggingFace hub

    Args:
        model (str): model name to be loaded
        device (str): device info

    Returns:
        object: Pipeline-based Named Entity Recognition model

    """
    logging.info("Loading Named Entity Recognition Pipeline...")

    try:
        ner = pipeline(
            task="ner",
            model=model,
            tokenizer=model,
            ignore_labels=[],
            framework="pt",
            device=-1 if device == "cpu" else 0,
        )
    except (HTTPError, OSError):
        logging.warning("Input model is not supported by HuggingFace Hub")
        raise

    def extract_entities_hf(sentences: List[str]):
        result = []
        total_entities = ner(sentences)

        if isinstance(total_entities[0], dict):
            total_entities = [total_entities]

        for line_entities in total_entities:
            result.append(grouped_entities(line_entities))

        return result

    return extract_entities_hf


def load_rel(model: str, device: str):
    """
    Load LUKE for Relation Extraction model and return its applicable function

    Args:
        model (str): model name to be loaded
        device (str): device info

    Returns:
        function: LUKE-based Relation Extraction function

    """
    logging.info("Loading Relation Extraction Pipeline...")

    try:
        # yapf:disable
        tokenizer = LukeTokenizer.from_pretrained(model)
        model = LukeForEntityPairClassification.from_pretrained(model).to(device)
        # yapf:enable
    except (HTTPError, OSError):
        logging.warning("Input model is not supported by HuggingFace Hub")

    def extract_relation(sentences: List) -> List[Tuple]:
        """
        Extraction Relation based on Entity Information

        Args:
            sentence (str): original sentence containing context
            head_entity (Dict): head entity containing position information
            tail_entity (Dict): tail entity containing position information

        Returns:
            List[Tuple]: list of (head_entity, relation, tail_entity) formatted triples

        """
        triples = []

        # TODO: batchify
        for sentence in sentences:
            tokens = tokenizer(
                sentence["text"],
                entity_spans=[
                    (sentence["spans"][0][0], sentence["spans"][0][-1]),
                    (sentence["spans"][-1][0], sentence["spans"][-1][-1]),
                ],
                return_tensors="pt",
            ).to(device)
            outputs = model(**tokens)
            predicted_id = int(outputs.logits[0].argmax())
            relation = model.config.id2label[predicted_id]

            if relation != "no_relation":
                triples.append((
                    sentence["text"]
                    [sentence["spans"][0][0]:sentence["spans"][0][-1]],
                    relation,
                    sentence["text"]
                    [sentence["spans"][-1][0]:sentence["spans"][-1][-1]],
                ))

        return triples

    return extract_relation


def load_ie():
    """
    Load Stanford Open IE based Triple Extractor function

    Returns:
        function: Open IE annotate function to extract fact triple

    """
    logging.info("Loading Open IE Pipeline...")
    from openie import StanfordOpenIE
    client = StanfordOpenIE()
    return client.annotate
