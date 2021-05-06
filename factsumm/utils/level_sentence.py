from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_qg(model: str):
    print("Loading Question Generation Pipeline...")

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)

    def generate_question():
        # See also https://github.com/AMontgomerie/question_generator
        pass

    return generate_question


# TODO: NLI, FactCC
