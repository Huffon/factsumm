from setuptools import setup, find_packages

requirements = [
    "transformers==4.36.0",
    "pysbd==0.3.4",
    "bert-score==0.3.12",
    "rich==13.7.0",
    "protobuf==4.25.1",
    "sentencepiece==0.1.99",
    "sumeval==0.2.2",
]

VERSION = {}
with open("factsumm/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="factsumm",
    version=VERSION["__version__"],
    description=
    "FactSumm: Factual Consistency Scorer for Abstractive Summarization",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    url="https://github.com/huffon/factsumm",
    author="Hoon Heo",
    license="Apache 2.0",
    packages=find_packages(include=["factsumm", "factsumm.*"]),
    install_requires=requirements,
    python_requires=">=3.8.0",
    extras_require={},
)
