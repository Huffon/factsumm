# FactSumm: Factual Consistency Scorer for Abstractive Summarization

`FactSumm` is a toolkit that scores *__Factualy Consistency__* for **Abstract Summarization**

Without fine-tuning, you can simply apply a variety of downstream tasks to both `the source article` and `the generated abstractive summary`

![](assets/triples.png)

For example, by extracting **fact triples** from source articles and generated summaries, we can verify that generated summaries correctly reflect source-based facts ( *See image above* )

As you can guess, this *PoC-ish* project uses a lot of pre-trained modules that require __*super-duper*__ computing resources

So don't blame me, just take it as a concept project 👀

<br>

## Installation

`FactSumm` requires *Java* to be installed in your environment to use **Stanford OpenIE**. With *Java* and *Python 3*, you can install `FactSumm` from source repository:

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
>>> article = "Superman is a fictional superhero who first appeared in American comic books published by DC Comics. The character was created by writer Jerry Siegel and artist Joe Shuster, and first appeared in the comic book Action Comics #1. Superman has been adapted to a number of other media which includes radio serials, novels, movies, television shows and theatre. Although Superman was not the first superhero character, he popularized the superhero archetype and established its conventions. Superheroes are usually judged by how closely they resemble the standard set by Superman. He was the best-selling superhero character in American comic books up until the 1980s."
>>> summary = "Superman is a fictional superhero who first appeared in American comic books published by Marvel Comics. The character was created by writer Jerry Siegel and artist Joe Shuster. He popularized the superhero archetype and established its conventions. Superman has been adapted to a number of other media which includes radio serials, novels, movies, television shows and theatre."
>>> factsumm(article, summary, verbose=True)
SOURCE Entities
1: [('Superman', 'PER'), ('American', 'MISC'), ('DC Comics', 'ORG')]
2: [('Jerry Siegel', 'PER'), ('Joe Shuster', 'PER'), ('Action Comics', 'MISC')]
3: [('Superman', 'PER')]
4: [('Superman', 'PER')]
5: [('Superman', 'PER')]
6: [('American', 'MISC')]

SUMMARY Entities
1: [('Superman', 'PER'), ('American', 'MISC'), ('Marvel Comics', 'ORG')]
2: [('Jerry Siegel', 'PER'), ('Joe Shuster', 'PER')]
3: []
4: [('Superman', 'PER')]

SOURCE Facts
('American', 'per:alternate_names', 'Superman')
('American', 'per:employee_of', 'DC Comics')
('DC Comics', 'org:country_of_headquarters', 'American')
('Superman', 'per:employee_of', 'DC Comics')
('Superman', 'per:origin', 'American')

SUMMARY Facts
('American', 'per:alternate_names', 'Superman')
('Marvel Comics', 'org:country_of_headquarters', 'American')
('Superman', 'per:employee_of', 'Marvel Comics')
('American', 'per:employee_of', 'Marvel Comics')
('Superman', 'per:origin', 'American')

COMMON Facts
('American', 'per:alternate_names', 'Superman')
('Superman', 'per:origin', 'American')

DIFF Facts
('Marvel Comics', 'org:country_of_headquarters', 'American')
('Superman', 'per:employee_of', 'Marvel Comics')
('American', 'per:employee_of', 'Marvel Comics')

Fact Score: 0.4

Answers based on SOURCE (Questions are generated from Summary)
[Q] What is the name of the fictional superhero that first appeared in comic books?     [Ent] Superman  [Pred] Superman
[Q] In what country did Superman first appear?  [Ent] American  [Pred] American
[Q] What company published the first Superman comic book?       [Ent] Marvel Comics     [Pred] DC Comics
[Q] Who created the character?  [Ent] Jerry Siegel      [Pred] Jerry Siegel and artist Joe Shuster
[Q] Who created the character?  [Ent] Joe Shuster       [Pred] Jerry Siegel and artist Joe Shuster
[Q] What superhero has been adapted to a number of other media? [Ent] Superman  [Pred] Superman

Answers based on SUMMARY (Questions are generated from Summary)
[Q] What is the name of the fictional superhero that first appeared in comic books?     [Ent] Superman  [Pred] Superman
[Q] In what country did Superman first appear?  [Ent] American  [Pred] American
[Q] What company published the first Superman comic book?       [Ent] Marvel Comics     [Pred] Marvel Comics
[Q] Who created the character?  [Ent] Jerry Siegel      [Pred] Jerry Siegel and artist Joe Shuster
[Q] Who created the character?  [Ent] Joe Shuster       [Pred] Jerry Siegel and artist Joe Shuster
[Q] What superhero has been adapted to a number of other media? [Ent] Superman  [Pred] Superman

QAGS Score: 0.9166666666666666

SOURCE Triples
('they', 'closely resemble', 'standard set')
('He', 'was', 'best selling character')
('He', 'was', 'best selling character in comic books up until 1980s')
('He', 'was', 'superhero character in comic books up until 1980s')
('Superman', 'is fictional superhero', 'appeared in books published by DC Comics')
('he', 'established', 'its conventions')
...

SUMMARY Triples
('Superman', 'is fictional superhero', 'first appeared in American books published by Marvel Comics')
('Superman', 'is fictional superhero', 'first appeared in American comic books published by Marvel Comics')
('He', 'established', 'its conventions')
('Superman', 'is fictional superhero', 'first appeared in American books')
('Superman', 'is fictional superhero', 'first appeared in American comic books published')
...

Triple Score: 0.6774193548387096

Avg. ROUGE-1: 0.34586498627159923
Avg. ROUGE-2: 0.24065908743388897
Avg. ROUGE-L: 0.30456185003002245
```

<br>

## Sub-modules

From [here](https://arxiv.org/pdf/2104.14839.pdf), you can find various way to score **Factual Consistency level** with *Unsupervised methods*

<br>

### Triple-based Module


The triple-based module counts the overlap of fact triples between the generated summary and the source document.

<br>

### QA-based Module

![](assets/qa.png)

If you ask questions about the summary and the source document, you will get a similar answer if the summary realistically matches the source document

<br>

### OpenIE-based Module

Stanford OpenIE can extract relationships from raw strings. But it's important to note that it's based on the open scheme, not the closed scheme (like `Triple-based Module`).

For example, from `"Obama was born in Hawaii"`, OpenIE extracts (Obama, born in Hawaii). However, from `"Hawaii is the birthplace of Obama"`, it extracts (Hawaii, is the birthplace of, Obama). In common sense, the triples extracted from the two sentences should be identical, but OpenIE can't recognize that they are the same since it is based on an open scheme.

So the score for this module may be unstable

<br>

### ROUGE-based Module

Simple but effective word-level overlap ROUGE score

<br>

## References

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [BERTScore](https://github.com/Tiiiger/bert_score)
- [PySBD](https://github.com/nipunsadvilkar/pySBD)
- [The Factual Inconsistency Problem in Abstractive Text Summarization: A Survey](https://arxiv.org/abs/2104.14839.pdf)
- [Assessing The Factual Accuracy of Generated Text](https://arxiv.org/abs/1905.13322.pdf)
- [Asking and Answering Questions to Evaluate the Factual Consistency of Summaries](https://arxiv.org/abs/2004.04228)
- [FEQA: A Question Answering Evaluation Framework for Faithfulness Assessment in Abstractive Summarization](https://arxiv.org/abs/2005.03754)
