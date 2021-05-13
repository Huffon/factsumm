import unittest

from transformers import pipelines

from factsumm.utils.utils import Config, grouped_entities


class TestUtils(unittest.TestCase):

    def test_grouped_entities(self):
        config = Config()

        ner = pipelines(
            task="ner",
            model=config.NER_MODEL,
            tokenizer=config.NER_MODEL,
            ignore_labels=[],
            framework="pt",
        )

        article = "John Winston Ono Lennon was an English singer, songwriter, musician and peace activist who achieved worldwide fame as the founder, co-lead vocalist, and rhythm guitarist of the Beatles. His songwriting partnership with Paul McCartney remains the most successful in history. In 1969, he started the Plastic Ono Band with his second wife, Yoko Ono. After the Beatles disbanded in 1970, Lennon continued his career as a solo artist and as Ono's collaborator. Born in Liverpool, Lennon became involved in the skiffle craze as a teenager. In 1956, he formed his first band, the Quarrymen, which evolved into the Beatles in 1960. He was initially the group's de facto leader, a role gradually ceded to McCartney. Lennon was characterised for the rebellious nature and acerbic wit in his music, writing, drawings, on film and in interviews. In the mid-1960s, he had two books published: In His Own Write and A Spaniard in the Works, both collections of nonsense writings and line drawings. Starting with 1967's 'All You Need Is Love', his songs were adopted as anthems by the anti-war movement and the larger counterculture."
        entities = ner(article)

        grouped = grouped_entities(entities)
