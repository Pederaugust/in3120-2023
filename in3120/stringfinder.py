#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Iterator, Dict, Any
from .tokenizer import Tokenizer
from .trie import Trie


class StringFinder:
    """
    Given a trie encoding a dictionary of strings, efficiently finds the subset of strings in the dictionary
    that are also present in a given text buffer. I.e., in a sense computes the "intersection" or "overlap"
    between the dictionary and the text buffer.

    Uses a trie-walk algorithm similar to the Aho-Corasick algorithm with some simplifications and some minor
    NLP extensions. The running time of this algorithm is virtually independent of the size of the dictionary,
    and linear in the length of the buffer we are searching in.

    The tokenizer we use when scanning the input buffer is assumed to be the same as the one that was used
    when adding strings to the trie.
    """

    def __init__(self, trie: Trie, tokenizer: Tokenizer):
        self.__trie = trie  # Methods: add, consume, is_final
        self.__tokenizer = tokenizer

    def scan(self, buffer: str) -> Iterator[Dict[str, Any]]:
        """
        Scans the given buffer and finds all dictionary entries in the trie that are also present in the
        buffer. We only consider matches that begin and end on token boundaries.

        The matching dictionary entries, if any, are yielded back to the client as dictionaries having the
        keys "match" (str) and "range" (Tuple[int, int]).

        In a serious application we'd add more lookup/evaluation features, e.g., support for prefix matching,
        support for leftmost-longest matching (instead of reporting all matches), and support for lemmatization
        or similar linguistic variations.
        """

        for (token, rng) in self.__tokenizer.tokens(buffer):  # token = string, rng = tuple
            # First node might be a match or not
            trie = self.__trie.consume(token)
            if trie is not None and trie.is_final():
                if trie.keys() == [""]:
                    yield {"match": token, "range": rng}
