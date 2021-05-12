import unittest

from factsumm import FactSumm
from factsumm.utils.utils import Config, load_summarizer


class TestFactSum(unittest.TestCase):

    def test_whole_pipes(self):
        config = Config()

        factsumm = FactSumm()
        summarzier = load_summarizer(config.SUMM_MODEL)

        article = "John Winston Ono Lennon was an English singer, songwriter, musician and peace activist who achieved worldwide fame as the founder, co-lead vocalist, and rhythm guitarist of the Beatles. His songwriting partnership with Paul McCartney remains the most successful in history. In 1969, he started the Plastic Ono Band with his second wife, Yoko Ono. After the Beatles disbanded in 1970, Lennon continued his career as a solo artist and as Ono's collaborator. Born in Liverpool, Lennon became involved in the skiffle craze as a teenager. In 1956, he formed his first band, the Quarrymen, which evolved into the Beatles in 1960. He was initially the group's de facto leader, a role gradually ceded to McCartney. Lennon was characterised for the rebellious nature and acerbic wit in his music, writing, drawings, on film and in interviews. In the mid-1960s, he had two books published: In His Own Write and A Spaniard in the Works, both collections of nonsense writings and line drawings. Starting with 1967's 'All You Need Is Love', his songs were adopted as anthems by the anti-war movement and the larger counterculture."
        summary = summarzier(article)

        factsumm(article, summary)


if __name__ == "__main__":
    config = Config()

    factsumm = FactSumm()
    # summarzier = load_summarizer(config.SUMM_MODEL)

    article = "Lionel Andrés Messi (born 24 June 1987) is an Argentine professional footballer who plays as a forward and captains both Spanish club Barcelona and the Argentina national team. Often considered as the best player in the world and widely regarded as one of the greatest players of all time, Messi has won a record six Ballon d'Or awards, a record six European Golden Shoes, and in 2020 was named to the Ballon d'Or Dream Team."

    # summary = summarzier(article)[0]["summary_text"].strip()
    summary = "Lionel Andrés Messi (born 24 Aug 1997) is an Spanish professional footballer who plays as a forward and captains both Spanish club Barcelona and the Spanish national team."

    factsumm(article, summary, verbose=True)
