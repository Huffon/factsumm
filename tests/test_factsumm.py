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

    article = "Superman is a fictional superhero who first appeared in American comic books published by DC Comics. The character was created by writer Jerry Siegel and artist Joe Shuster, and first appeared in the comic book Action Comics #1. Superman has been adapted to a number of other media which includes radio serials, novels, movies, television shows and theatre. Although Superman was not the first superhero character, he popularized the superhero archetype and established its conventions. Superheroes are usually judged by how closely they resemble the standard set by Superman. He was the best-selling superhero character in American comic books up until the 1980s."
    # summary = summarzier(article)[0]["summary_text"].strip()
    summary = "Superman is a fictional superhero who first appeared in American comic books published by Marvel Comics. The character was created by writer Jerry Siegel and artist Joe Shuster. He popularized the superhero archetype and established its conventions. Superman has been adapted to a number of other media which includes radio serials, novels, movies, television shows and theatre."

    factsumm(article, summary, verbose=True)
