from typing import List, Tuple

from flair.data import Sentence
from flair.models import SequenceTagger
from rich import print
from transformers import LukeForEntityPairClassification, LukeTokenizer, pipeline

from factsumm.utils.utils import grouped_entities


def load_ner(model: str) -> object:
    """
    Load Named Entity Recognition model from HuggingFace hub

    Args:
        model (str): model name to be loaded

    Returns:
        object: Pipeline-based Named Entity Recognition model

    """
    print("Loading Named Entity Recognition Pipeline...")

    if "flair" in model:
        ner = SequenceTagger.load(model)

        def extract_entities(sentences: List[str]):
            result = list()

            for sentence in sentences:
                sentence = Sentence(sentence)
                ner.predict(sentence)
                line_result = sentence.to_dict(tag_type="ner")

                cache = dict()
                dedup = list()

                for entity in line_result["entities"]:
                    if entity["text"] not in cache:
                        dedup.append({
                            "word": entity["text"],
                            "entity": entity["labels"][0].value,
                            "start": entity["start_pos"],
                            "end": entity["end_pos"],
                        })
                        cache[entity["text"]] = None
                result.append(dedup)

            return result
    else:
        ner = pipeline(
            task="ner",
            model=model,
            tokenizer=model,
            ignore_labels=[],
            framework="pt",
        )

        def extract_entities(sentences: List[str]):
            result = list()
            total_entities = ner(sentences)

            if isinstance(total_entities[0], dict):
                total_entities = [total_entities]

            for line_entities in total_entities:
                result.append(grouped_entities(line_entities))

            return result

    return extract_entities


def load_rel(model: str):
    """
    Load LUKE for Relation Extraction model and return its applicable function

    Args:
        model (str): model name to be loaded

    Returns:
        function: LUKE-based Relation Extraction function

    """
    print("Loading Relation Extraction Pipeline...")

    tokenizer = LukeTokenizer.from_pretrained(model)
    model = LukeForEntityPairClassification.from_pretrained(model)

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
        triples = list()

        # TODO: batchify
        for sentence in sentences:
            tokens = tokenizer(
                sentence["text"],
                entity_spans=[
                    (sentence["spans"][0][0], sentence["spans"][0][-1]),
                    (sentence["spans"][-1][0], sentence["spans"][-1][-1]),
                ],
                return_tensors="pt",
            )
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
    print("Loading Open IE Pipeline...")
    from openie import StanfordOpenIE
    client = StanfordOpenIE()
    return client.annotate
