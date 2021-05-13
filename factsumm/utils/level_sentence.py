from typing import List

from bert_score import BERTScorer
from rich import print
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def load_qg(model: str):
    """
    Load Question Generation model from HuggingFace hub

    Args:
        model (str): model name to be loaded

    Returns:
        function: question generation function

    """
    print("Loading Question Generation Pipeline...")

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)

    def generate_question(sentences: List[str], total_entities: List):
        """
        Generation question using context and entity information

        Args:
            sentences (List[str]): list of sentences
            total_entities (List): list of entities

        Returns:
            List[Dict] list of question and answer (entity) pairs

        """
        qa_pairs = list()

        for sentence, line_entities in zip(sentences, total_entities):
            for entity in line_entities:
                entity = entity["word"]

                template = f"answer: {entity}  context: {sentence} </s>"

                # TODO: batchify
                tokens = tokenizer(
                    template,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )

                outputs = model.generate(**tokens, max_length=64)

                question = tokenizer.decode(outputs[0])
                question = question.replace("</s>", "")
                question = question.replace("<pad> question: ", "")

                qa_pairs.append({
                    "question": question,
                    "answer": entity,
                })

        return qa_pairs

    return generate_question


def load_qa(model: str):
    """
    Load Question Answering model from HuggingFace hub

    Args:
        model (str): model name to be loaded

    Returns:
        function: question answering function

    """
    print("Loading Question Answering Pipeline...")

    qa = pipeline(
        "question-answering",
        model=model,
        tokenizer=model,
        framework="pt",
    )

    def answer_question(context: str, qa_pairs: List):
        """
        Answer question via Span Prediction

        Args:
            context (str): context to be encoded
            qa_pairs (List): Question & Answer pairs generated from Question Generation pipe

        """
        answers = list()
        for qa_pair in qa_pairs:
            pred = qa(
                question=qa_pair["question"],
                context=context,
                handle_impossible_answer=True,
            )["answer"]
            answers.append({
                "question": qa_pair["question"],
                "answer": qa_pair["answer"],
                "prediction": pred if pred != "" else "<unanswerable>"
            })
        return answers

    return answer_question


def load_bert_score(model: str):
    """
    Load BERTScore model from HuggingFace hub

    Args:
        model (str): model name to be loaded

    Returns:
        function: BERTScore score function

    """
    print("Loading BERTScore Pipeline...")

    scorer = BERTScorer(model_type=model, lang="en", rescale_with_baseline=True)
    return scorer.score
