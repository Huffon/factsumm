from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def load_qg(model: str):
    print("Loading Question Generation Pipeline...")

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)

    def generate_question(sentences: List[str], total_entities: List):
        qa_pairs = list()

        if isinstance(total_entities[0], dict):
            total_entities = [total_entities]

        for sentence, line_entities in zip(sentences, total_entities):
            for entity in line_entities:
                entity = entity['word']

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
                question = question.replace("<pad> questions: ", "")

                qa_pairs.append({
                    "question": question,
                    "answer": entity,
                })

        return qa_pairs

    return generate_question


def load_qa(model: str):
    print("Loading Question Answering Pipeline...")

    qa = pipeline(
        "question-answering",
        model=model,
        tokenizer=model,
        framework="pt",
    )

    def answer_question(context: str, qa_pairs: List):
        for qa_pair in qa_pairs:
            answer = qa(question=qa_pair["question"], context=context)

    return answer_question


# TODO: NLI, FactCC
