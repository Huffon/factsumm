# FactSumm: Factual Consistency Scorer for Abstractive Summarization

`FactSumm` is a toolkit that scores Factualy Consistency for Abstract Summarization model

Without fine-tuning the data, you can simply apply a variety of downstream tasks to both the original document and the generated summary

![](assets/triples.png)

For example, by extracting fact triples from source documents and generated summaries, we can verify that generated summaries correctly reflect source-based facts ( *See image above* )

As you can guess, this PoC-ish project uses a lot of Pre-trained modules that require __*super-duper*__ computing resources

So don't blame me, just take it as a fun concept project 👀

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

- [ ] Add Question-based score module
- [ ] Add NLI-based score module

<br>

## References

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [BERTScore](https://github.com/Tiiiger/bert_score)
- [PySBD](https://github.com/nipunsadvilkar/pySBD)
- [The Factual Inconsistency Problem in Abstractive Text Summarization: A Survey](https://arxiv.org/pdf/2104.14839.pdf)
