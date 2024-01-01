import logging

from bert_score import BERTScorer


def load_bert_score(device: str):
    """
    Load BERTScore model from HuggingFace hub

    Args:
        model (str): model name to be loaded
        device (str): device info

    Returns:
        function: BERTScore score function

    """
    logging.debug("Loading BERTScore Pipeline...")

    try:
        scorer = BERTScorer(
            lang="en",
            device=device,
        )
        return scorer.score
    except KeyError:
        logging.warning("Input model is not supported by BERTScore")
