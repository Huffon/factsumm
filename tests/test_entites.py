import unittest

from factsumm import FactSumm


class TestEntities(unittest.TestCase):

    def test_extract_facts(self):
        factsumm = FactSumm()

        article = "Lionel Andrés Messi (born 24 June 1987) is an Argentine professional footballer who plays as a forward and captains both Spanish club Barcelona and the Argentina national team. Often considered as the best player in the world and widely regarded as one of the greatest players of all time, Messi has won a record six Ballon d'Or awards, a record six European Golden Shoes, and in 2020 was named to the Ballon d'Or Dream Team."
        summary = "Lionel Andrés Messi (born 24 Aug 1997) is an Spanish professional footballer who plays as a forward and captains both Spanish club Barcelona and the Spanish national team."

        factsumm.extract_facts(article, summary, verbose=True)


if __name__ == "__main__":
    unittest.main()
