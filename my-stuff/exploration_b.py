
from context import in3120
import unittest
from in3120.corpus import Corpus
from in3120.normalizer import Normalizer

from in3120.tokenizer import Tokenizer


class TestSuffixArray(unittest.TestCase):
    def setUp(self):
        self.normalizer: Normalizer = in3120.SimpleNormalizer()
        self.tokenizer: Tokenizer = in3120.SimpleTokenizer()
        self.corpus: Corpus = in3120.InMemoryCorpus()
        self.corpus.add_document(in3120.InMemoryDocument(
            self.corpus.size(), {"a": "Japanese リンク"}))
        self.corpus.add_document(in3120.InMemoryDocument(
            self.corpus.size(), {"a": "Cedilla \u0043\u0327 and \u00C7 foo"}))

        self.engine = in3120.SuffixArray(
            self.corpus, ["a"], self.normalizer, self.tokenizer)

    def test_build_suffix_array_initializes_haystack(self):
        self.assertEqual(self.engine._SuffixArray__haystack, [
            (0, self.normalizer.normalize("japanese リンク")),
            (1, self.normalizer.normalize("Cedilla \u00C7 and \u00C7 foo"))
        ], "The haystack was not implemented correctly")

    def test_build_suffix_array_initializes_suffixes(self):
        self.assertListEqual(self.engine._SuffixArray__suffixes, [
            (1, 10),
            (1, 0),
            (1, 16),
            (0, 0),
            (1, 8),
            (1, 14),
            (0, 9),
        ], "The suffixes were not implemented correctly, check sorting or tokenization/normalization")

    def test_suffix_of_doc(self):
        haystack = [
            (0, "japanese リンク"),
            (1, "Cedilla \u0043\u0327 and \u00C7 foo")
        ]
        suffixes = [
            (1, 10),
            (1, 0),
            (1, 8),
            (0, 0),
            (1, 16),
            (1, 14),
            (0, 9),
        ]
        sf_array = in3120.SuffixArray(
            self.corpus, ["a"], self.normalizer, self.tokenizer)
        sf_array._SuffixArray__haystack = haystack
        sf_array._SuffixArray__suffixes = suffixes
        self.assertEqual(sf_array._SuffixArray__suffix_of_doc(
            (0, 0)), "japanese リンク")
        self.assertEqual(sf_array._SuffixArray__suffix_of_doc(
            (1, 0)), "Cedilla \u0043\u0327 and \u00C7 foo")

    def test_first_query(self):

        matches = list(self.engine.evaluate(
            "ﾘﾝｸ", {"debug": False, "hit_count": 5}))
        self.assertEqual(matches[0]["document"].document_id, 0)
        self.assertEqual(matches[0]["score"], 1)

    def test_second_query(self):
        matches = list(self.engine.evaluate(
            "\u00C7", {"debug": False, "hit_count": 5}))
        self.assertEqual(matches[0]["document"].document_id, 1)
        # Should match two tokens in document 1, therefore score 2
        self.assertEqual(matches[0]["score"], 2)


class TestStringFinder(unittest.TestCase):
    def setUp(self):
        self.tokenizer = in3120.SimpleTokenizer()
        self.trie = in3120.Trie()
        self.finder = in3120.StringFinder(self.trie, self.tokenizer)

    def test_something(self):
        print("Not implemented")


if __name__ == '__main__':
    unittest.main()
