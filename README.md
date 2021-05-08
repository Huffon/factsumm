# FactSumm: Factual Consistency Scorer for Abstractive Summarization

`FactSumm` is a toolkit that scores *__Factualy Consistency__* for **Abstract Summarization**

Without fine-tuning, you can simply apply a variety of downstream tasks to both `the source article` and `the generated abstractive summary`

![](assets/triples.png)

For example, by extracting **fact triples** from source articles and generated summaries, we can verify that generated summaries correctly reflect source-based facts ( *See image above* )

As you can guess, this *PoC-ish* project uses a lot of pre-trained modules that require __*super-duper*__ computing resources

So don't blame me, just take it as a concept project 👀

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
pip install .
```

<br>

## Usage

```python
>>> from factsumm import FactSumm
>>> scorer = FactSumm()
>>> article = "Steven Paul Jobs was an American business magnate, industrial designer, investor, and media proprietor. He was the chairman, chief executive officer, and co-founder of Apple Inc.; the chairman and majority shareholder of Pixar; a member of The Walt Disney Company's board of directors following its acquisition of Pixar; and the founder, chairman, and CEO of NeXT. Jobs is widely recognized as a pioneer of the personal computer revolution of the 1970s and 1980s, along with his early business partner and fellow Apple co-founder Steve Wozniak."
>>> summary = "Steven Paul Jobs was the chairman, chief executive officer, and co-founder of Apple Inc. He is widely recognized as a pioneer of the personal computer revolution of the 1970s and 1980s . Jobs is the chairman of The Walt Disney Company's board of directors following its acquisition of Pixar ."
>>> scorer(article, summary) 
```

<br>

## Sub-modules

From [here](https://arxiv.org/pdf/2104.14839.pdf), you can find various way to score **Factual Consistency level** with *Unsupervised methods*

<br>

### Triple-based Factual Consistency

count the fact overlap between generated summary and the source document

not combination, but permutation

<br>

### QA-based Factual Consistency

![](assets/qa.png)

If you ask questions about the summary and the source document, you will get a similar answer if the summary realistically matches the source document

<br>

## References

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [BERTScore](https://github.com/Tiiiger/bert_score)
- [PySBD](https://github.com/nipunsadvilkar/pySBD)
- [The Factual Inconsistency Problem in Abstractive Text Summarization: A Survey](https://arxiv.org/pdf/2104.14839.pdf)
