import unittest

from factsumm import FactSumm
from factsumm.utils.utils import Config, load_summarizer


class TestFactSum(unittest.TestCase):

    def test_whole_pipes(self):
        config = Config()

        factsumm = FactSumm()
        summarzier = load_summarizer(config.SUMM_MODEL)

        article = "Steven Paul Jobs was an American business magnate, industrial designer, investor, and media proprietor. He was the chairman, chief executive officer, and co-founder of Apple Inc.; the chairman and majority shareholder of Pixar; a member of The Walt Disney Company's board of directors following its acquisition of Pixar; and the founder, chairman, and CEO of NeXT. Jobs is widely recognized as a pioneer of the personal computer revolution of the 1970s and 1980s, along with his early business partner and fellow Apple co-founder Steve Wozniak."
        summary = summarzier(article)

        factsumm(article, summary)


if __name__ == "__main__":
    config = Config()

    factsumm = FactSumm()
    summarzier = load_summarizer(config.SUMM_MODEL)

    article = "Steven Paul Jobs was an American business magnate, industrial designer, investor, and media proprietor. He was the chairman, chief executive officer, and co-founder of Apple Inc.; the chairman and majority shareholder of Pixar; a member of The Walt Disney Company's board of directors following its acquisition of Pixar; and the founder, chairman, and CEO of NeXT. Jobs is widely recognized as a pioneer of the personal computer revolution of the 1970s and 1980s, along with his early business partner and fellow Apple co-founder Steve Wozniak."
    summary = summarzier(article)[0]["summary_text"].strip()
    print(summary)

    factsumm(article, summary)
