# FactSumm: Factual Consistency Scorer for Abstractive Summarization

<p align="center">
  <a href="https://github.com/huffon/factsumm/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huffon/factsumm.svg" /></a>
  <a href="https://github.com/huffon/factsumm/blob/master/LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" /></a>
  <a href="https://github.com/huffon/factsumm/issues"><img alt="Issues" src="https://img.shields.io/github/issues/huffon/factsumm" /></a>
</p>

`FactSumm` is a toolkit that scores _**Factualy Consistency**_ for **Abstract Summarization**

Without fine-tuning, you can simply apply a variety of downstream tasks to both `the source article` and `the generated abstractive summary`

![](assets/triples.png)

For example, by extracting **fact triples** from source articles and generated summaries, you can verify that generated summaries correctly reflect source-based facts ( _See image above_ )

<br>

## Installation

With and _Python 3_ (>= 3.8), you can install `factsumm` simply using `pip`:

```bash
pip install factsumm
```

Or you can install `FactSumm` from source repository:

```bash
git clone https://github.com/huffon/factsumm
cd factsumm
pip install .
```

<br>

## Usage

```python
>>> from factsumm import FactSumm
>>> factsumm = FactSumm()
>>> article = "Lionel Andrés Messi (born 24 June 1987) is an Argentine professional footballer who plays as a forward and captains both Spanish club Barcelona and the Argentina national team. Often considered as the best player in the world and widely regarded as one of the greatest players of all time, Messi has won a record six Ballon d'Or awards, a record six European Golden Shoes, and in 2020 was named to the Ballon d'Or Dream Team."
>>> summary = "Lionel Andrés Messi (born 24 Aug 1997) is an Spanish professional footballer who plays as a forward and captains both Spanish club Barcelona and the Spanish national team."
>>> factsumm(article, summary, verbose=True)
<Source Entities>
Line No.1: [[('Lionel Andrés Messi', 'PERSON'), ('24 June 1987', 'DATE'), ('Argentine', 'NORP'), ('Spanish', 'NORP'), ('Barcelona', 'ORG'), ('Argentina', 'GPE')]]
Line No.2: [[('one', 'CARDINAL'), ('Messi', 'PERSON'), ('six', 'CARDINAL'), ("Ballon d'Or", 'WORK_OF_ART'), ('European Golden Shoes', 'WORK_OF_ART'), ('2020', 'DATE'), ("Ballon d'Or Dream Team", 'WORK_OF_ART')]]

<Summary Entities>
Line No.1: [[('Lionel Andrés Messi', 'PERSON'), ('24 Aug 1997', 'DATE'), ('Spanish', 'NORP'), ('Barcelona', 'ORG')]]

<Source Facts>
('Lionel Andrés Messi', 'per:origin', 'Argentine')
('Lionel Andrés Messi', 'per:employee_of', 'Barcelona')
('Barcelona', 'org:country_of_headquarters', 'Spanish')
('Lionel Andrés Messi', 'per:date_of_birth', '24 June 1987')
('Lionel Andrés Messi', 'per:countries_of_residence', 'Argentina')
('Spanish', 'org:top_members/employees', 'Lionel Andrés Messi')
('Barcelona', 'org:top_members/employees', 'Lionel Andrés Messi')

<Summary Facts>
('Lionel Andrés Messi', 'per:employee_of', 'Barcelona')
('Lionel Andrés Messi', 'per:date_of_birth', '24 Aug 1997')
('Barcelona', 'org:country_of_headquarters', 'Spanish')
('Spanish', 'org:top_members/employees', 'Lionel Andrés Messi')
('Barcelona', 'org:top_members/employees', 'Lionel Andrés Messi')
('Lionel Andrés Messi', 'per:origin', 'Spanish')
('Lionel Andrés Messi', 'per:countries_of_residence', 'Spanish')

<Common Facts>
('Barcelona', 'org:top_members/employees', 'Lionel Andrés Messi')
('Lionel Andrés Messi', 'per:employee_of', 'Barcelona')
('Barcelona', 'org:country_of_headquarters', 'Spanish')
('Spanish', 'org:top_members/employees', 'Lionel Andrés Messi')

<Diff Facts>
('Lionel Andrés Messi', 'per:date_of_birth', '24 Aug 1997')
('Lionel Andrés Messi', 'per:origin', 'Spanish')
('Lionel Andrés Messi', 'per:countries_of_residence', 'Spanish')

Fact Score: 0.5714285714285714
Answers based on Source (Questions are generated from Summary)
[Q] Who is the captain of the Spanish national team?	[Pred] <unanswerable>
[Q] When was Lionel Andrés Messi born?	[Pred] 24 June 1987
[Q] Lionel Andrés Messi is a professional footballer of what nationality?	[Pred] Argentine
[Q] Lionel Messi is a captain of which Spanish club?	[Pred] Barcelona

Answers based on Summary (Questions are generated from Summary)
[Q] Who is the captain of the Spanish national team?	[Pred] Lionel Andrés Messi
[Q] When was Lionel Andrés Messi born?	[Pred] 24 Aug 1997
[Q] Lionel Andrés Messi is a professional footballer of what nationality?	[Pred] Spanish
[Q] Lionel Messi is a captain of which Spanish club?	[Pred] Barcelona

QAGS Score: 0.3333333333333333

Avg. ROUGE-1: 0.4415584415584415
Avg. ROUGE-2: 0.3287671232876712
Avg. ROUGE-L: 0.4415584415584415
<BERTScore Score>
Precision: 0.9760397672653198
Recall: 0.9778039455413818
F1: 0.9769210815429688
```

You can use the GPU with the `device`. If you want to use GPU, pass `cuda` (default is `cpu`)

```python
>>> factsumm(article, summary, device="cuda")
```

<br>

## Sub-modules

From [here](https://arxiv.org/pdf/2104.14839.pdf), you can find various way to score **Factual Consistency level** with _Unsupervised methods_

<br>

### Triple-based Module ( _closed-scheme_ )

```python
>>> from factsumm import FactSumm
>>> factsumm = FactSumm()
>>> factsumm.extract_facts(article, summary, verbose=True)
<Source Entities>
Line No.1: [[('Lionel Andrés Messi', 'PERSON'), ('24 June 1987', 'DATE'), ('Argentine', 'NORP'), ('Spanish', 'NORP'), ('Barcelona', 'ORG'), ('Argentina', 'GPE')]]
Line No.2: [[('one', 'CARDINAL'), ('Messi', 'PERSON'), ('six', 'CARDINAL'), ("Ballon d'Or", 'WORK_OF_ART'), ('European Golden Shoes', 'WORK_OF_ART'), ('2020', 'DATE'), ("Ballon d'Or Dream Team", 'WORK_OF_ART')]]

<Summary Entities>
Line No.1: [[('Lionel Andrés Messi', 'PERSON'), ('24 Aug 1997', 'DATE'), ('Spanish', 'NORP'), ('Barcelona', 'ORG')]]

<Source Facts>
('Lionel Andrés Messi', 'per:origin', 'Argentine')
('Lionel Andrés Messi', 'per:employee_of', 'Barcelona')
('Barcelona', 'org:country_of_headquarters', 'Spanish')
('Lionel Andrés Messi', 'per:date_of_birth', '24 June 1987')
('Lionel Andrés Messi', 'per:countries_of_residence', 'Argentina')
('Spanish', 'org:top_members/employees', 'Lionel Andrés Messi')
('Barcelona', 'org:top_members/employees', 'Lionel Andrés Messi')

<Summary Facts>
('Lionel Andrés Messi', 'per:employee_of', 'Barcelona')
('Lionel Andrés Messi', 'per:date_of_birth', '24 Aug 1997')
('Barcelona', 'org:country_of_headquarters', 'Spanish')
('Spanish', 'org:top_members/employees', 'Lionel Andrés Messi')
('Barcelona', 'org:top_members/employees', 'Lionel Andrés Messi')
('Lionel Andrés Messi', 'per:origin', 'Spanish')
('Lionel Andrés Messi', 'per:countries_of_residence', 'Spanish')

<Common Facts>
('Barcelona', 'org:top_members/employees', 'Lionel Andrés Messi')
('Lionel Andrés Messi', 'per:employee_of', 'Barcelona')
('Barcelona', 'org:country_of_headquarters', 'Spanish')
('Spanish', 'org:top_members/employees', 'Lionel Andrés Messi')

<Diff Facts>
('Lionel Andrés Messi', 'per:date_of_birth', '24 Aug 1997')
('Lionel Andrés Messi', 'per:origin', 'Spanish')
('Lionel Andrés Messi', 'per:countries_of_residence', 'Spanish')

Fact Score: 0.5714285714285714
```

The triple-based module counts the overlap of fact triples between the generated summary and the source document.

<br>

### QA-based Module

![](assets/qa.png)

If you ask questions about the summary and the source document, you will get a similar answer if the summary realistically matches the source document

```python
>>> from factsumm import FactSumm
>>> factsumm = FactSumm()
>>> factsumm.extract_qas(article, summary, verbose=True)
Answers based on Source (Questions are generated from Summary)
[Q] Who is the captain of the Spanish national team?	[Pred] <unanswerable>
[Q] When was Lionel Andrés Messi born?	[Pred] 24 June 1987
[Q] Lionel Andrés Messi is a professional footballer of what nationality?	[Pred] Argentine
[Q] Lionel Messi is a captain of which Spanish club?	[Pred] Barcelona

Answers based on Summary (Questions are generated from Summary)
[Q] Who is the captain of the Spanish national team?	[Pred] Lionel Andrés Messi
[Q] When was Lionel Andrés Messi born?	[Pred] 24 Aug 1997
[Q] Lionel Andrés Messi is a professional footballer of what nationality?	[Pred] Spanish
[Q] Lionel Messi is a captain of which Spanish club?	[Pred] Barcelona

QAGS Score: 0.3333333333333333
```

<br>

### ROUGE-based Module

```python
>>> from factsumm import FactSumm
>>> factsumm = FactSumm()
>>> factsumm.calculate_rouge(article, summary)
Avg. ROUGE-1: 0.4415584415584415
Avg. ROUGE-2: 0.3287671232876712
Avg. ROUGE-L: 0.4415584415584415
```

Simple but effective word-level overlap ROUGE score

<br>

### BERTScore Module

```python
>>> from factsumm import FactSumm
>>> factsumm = FactSumm()
>>> factsumm.calculate_bert_score(article, summary)
<BERTScore Score>
Precision: 0.9760397672653198
Recall: 0.9778039455413818
F1: 0.9769210815429688
```

[BERTScore](https://github.com/Tiiiger/bert_score) can be used to calculate the similarity between each source sentence and the summary sentence

<br>

### Citation

If you apply this library to any project, please cite:

```
@misc{factsumm,
  author       = {Heo, Hoon},
  title        = {FactSumm: Factual Consistency Scorer for Abstractive Summarization},
  howpublished = {\url{https://github.com/Huffon/factsumm}},
  year         = {2021},
}
```

<br>

## References

- [The Factual Inconsistency Problem in Abstractive Text Summarization: A Survey](https://arxiv.org/abs/2104.14839.pdf)
- [Assessing The Factual Accuracy of Generated Text](https://arxiv.org/abs/1905.13322.pdf)
- [Asking and Answering Questions to Evaluate the Factual Consistency of Summaries](https://arxiv.org/abs/2004.04228)
- [FEQA: A Question Answering Evaluation Framework for Faithfulness Assessment in Abstractive Summarization](https://arxiv.org/abs/2005.03754)
