# FactSumm: Factual Consistency Scorer for Abstractive Summarization

`FactSumm` is Tool-kit to score Factual Consistency for Abstractive Summarization

<br>

## Installation

You can install `factsumm` simply using `pip`:

```bash
pip install factsumm
```

Or you can install it from source repository:

```bash
git clone https://github.com/huffon/factsumm
cd factsumm
pip install factsumm
```

<br>

## Usage

```python
>>> from factsumm import FactSumm
>>> scorer = FactSumm()
>>> source = ""
>>> summary = ""
>>> scorer(source, summary) 
```

<br>

## TODO

- [ ] Add NLI-based score module
- [ ] Add Question-based score module

<br>

## References

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [BERTScore](https://github.com/Tiiiger/bert_score)
- [PySBD](https://github.com/nipunsadvilkar/pySBD)
