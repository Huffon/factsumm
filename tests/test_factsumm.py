import unittest

from factsumm import FactSumm
from factsumm.utils.utils import Config, load_summarizer


class TestFactSum(unittest.TestCase):

    def test_whole_pipes(self):
        config = Config()

        factsumm = FactSumm()
        summarzier = load_summarizer(config.SUMM_MODEL)

        article = "Marie Curie was a Polish and French physicist and chemist who conducted pioneering research on radioactivity. As the first of the Curie family legacy of five Nobel Prizes, she was the first woman to win a Nobel Prize, the first person and the only woman to win the Nobel Prize twice, and the only person to win the Nobel Prize in two scientific fields. She was the first woman to become a professor at the University of Paris in 1906. She was born in Warsaw, in what was then the Kingdom of Poland, part of the Russian Empire. She studied at Warsaw's clandestine Flying University and began her practical scientific training in Warsaw. In 1891, aged 24, she followed her elder sister Bronisława to study in Paris, where she earned her higher degrees and conducted her subsequent scientific work. In 1895 she married the French physicist Pierre Curie, and she shared the 1903 Nobel Prize in Physics with him and with the physicist Henri Becquerel for their pioneering work developing the theory of 'radioactivity'—a term she coined. In 1906 Pierre Curie died in a Paris street accident. Marie won the 1911 Nobel Prize in Chemistry for her discovery of the elements polonium and radium, using techniques she invented for isolating radioactive isotopes."
        summary = summarzier(article)

        factsumm(article, summary)


if __name__ == "__main__":
    config = Config()

    factsumm = FactSumm()
    summarzier = load_summarizer(config.SUMM_MODEL)

    article = "Marie Curie was a Polish and French physicist and chemist who conducted pioneering research on radioactivity. As the first of the Curie family legacy of five Nobel Prizes, she was the first woman to win a Nobel Prize, the first person and the only woman to win the Nobel Prize twice, and the only person to win the Nobel Prize in two scientific fields. She was the first woman to become a professor at the University of Paris in 1906. She was born in Warsaw, in what was then the Kingdom of Poland, part of the Russian Empire. She studied at Warsaw's clandestine Flying University and began her practical scientific training in Warsaw. In 1891, aged 24, she followed her elder sister Bronisława to study in Paris, where she earned her higher degrees and conducted her subsequent scientific work. In 1895 she married the French physicist Pierre Curie, and she shared the 1903 Nobel Prize in Physics with him and with the physicist Henri Becquerel for their pioneering work developing the theory of 'radioactivity'—a term she coined. In 1906 Pierre Curie died in a Paris street accident. Marie won the 1911 Nobel Prize in Chemistry for her discovery of the elements polonium and radium, using techniques she invented for isolating radioactive isotopes."
    summary = summarzier(article)[0]["summary_text"].strip()
    print(summary)

    factsumm(article, summary)
