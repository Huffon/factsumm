from bert_score import BERTScorer
from rich import print


def load_bert_score(model: str, device: str):
    """
    Load BERTScore model from HuggingFace hub

    Args:
        model (str): model name to be loaded
        device (str): device info

    Returns:
        function: BERTScore score function

    """
    print("Loading BERTScore Pipeline...")

    try:
        scorer = BERTScorer(
            model_type=model,
            lang="en",
            rescale_with_baseline=True,
            device=device,
        )
        return scorer.score
    except KeyError:
        print("Input model is not supported by BERTScore")
