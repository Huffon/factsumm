from setuptools import setup, find_packages
requirements = [
    "transformers",
    "pysbd",
    "bert-score",
    "dataclasses; python_version<'3.7'",
    "rich",
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
    license="MIT",
    packages=find_packages(include=["factsumm", "factsumm.*"]),
    install_requires=requirements,
    python_requires=">=3.6.0",
    extras_require={},
)
