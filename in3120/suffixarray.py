#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterator, Iterable, Tuple, List
import math
from .corpus import Corpus
from .normalizer import Normalizer
from .tokenizer import Tokenizer


class SuffixArray:
    """
    A simple suffix array implementation. Allows us to conduct efficient substring searches.
    The prefix of a suffix is an infix!

    In a serious application we'd make use of least common prefixes (LCPs), pay more attention
    to memory usage, and add more lookup/evaluation features.
    """

    def __init__(self, corpus: Corpus, fields: Iterable[str], normalizer: Normalizer, tokenizer: Tokenizer):
        self.__corpus: Corpus = corpus
        self.__normalizer: Normalizer = normalizer
        self.__tokenizer: Tokenizer = tokenizer

        # The (<document identifier>, <searchable content>) pairs.
        self.__haystack: List[Tuple[int, str]] = []

        # The sorted (<haystack index>, <start offset>) pairs.
        self.__suffixes: List[Tuple[int, int]] = []

        # Construct the haystack and the suffix array itself.
        self.__build_suffix_array(fields)

    def __build_suffix_array(self, fields: Iterable[str]) -> None:
        """
        Builds a simple suffix array from the set of named fields in the document collection.
        The suffix array allows us to search across all named fields in one go.
        """
        self.__haystack = [(doc.document_id, self.__normalize(" ".join([doc[field] for field in fields])))
                           for doc in self.__corpus]
        self.__suffixes = [(idx, rngs[0]) for (
            idx, doc) in self.__haystack for rngs in self.__tokenizer.ranges(doc)]
        self.__suffixes.sort(key=lambda x: self.__suffix_of_doc(x))

    def __suffix_of_doc(self, suffix: Tuple[int, int]) -> str:
        return self.__haystack[self.__haystack_index(suffix)][1][self.__offset(suffix):]

    def __haystack_index(self, suffix: Tuple[int, int]) -> int:
        return suffix[0]

    def __offset(self, suffix: Tuple[int, int]) -> int:
        return suffix[1]

    def __normalize(self, buffer: str) -> str:
        """
        Produces a normalized version of the given string. Both queries and documents need to be
        identically processed for lookups to succeed.
        """
        return self.__normalizer.normalize(buffer)

    def __binary_search(self, needle: str) -> int:
        """
        Does a binary search for a given normalized query (the needle) in the suffix array (the haystack).
        Returns the position in the suffix array where the normalized query is either found, or, if not found,
        should have been inserted.

        Kind of silly to roll our own binary search instead of using the bisect module, but seems needed
        prior to Python 3.10 due to how we represent the suffixes via (index, offset) tuples. Version 3.10
        added support for specifying a key.
        """
        left = 0
        right = len(self.__suffixes) - 1
        while left <= right:
            middle = math.floor((left+right)/2)
            suffix = self.__suffix_of_doc(self.__suffixes[middle])
            if suffix < needle:
                left = middle + 1
            elif suffix > needle:
                right = middle - 1
            else:
                return middle
        return middle

    def __doc_id_from_suffix(self, suffix: Tuple[int, int]) -> int:
        return self.__haystack[self.__haystack_index(suffix)][0]

    def evaluate(self, query: str, options: dict) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing a "phrase prefix search".  E.g., for a supplied query phrase like
        "to the be", we return documents that contain phrases like "to the bearnaise", "to the best",
        "to the behemoth", and so on. I.e., we require that the query phrase starts on a token boundary in the
        document, but it doesn't necessarily have to end on one.

        The matching documents are ranked according to how many times the query substring occurs in the document,
        and only the "best" matches are yielded back to the client. Ties are resolved arbitrarily.

        The client can supply a dictionary of options that controls this query evaluation process: The maximum
        number of documents to return to the client is controlled via the "hit_count" (int) option.

        The results yielded back to the client are dictionaries having the keys "score" (int) and
        "document" (Document).
        """
        normalized_query = self.__normalize(query)
        print("Searching for", normalized_query)

        idx_of_needle = self.__binary_search(
            normalized_query)  # O(log n)
        print("Found at index", idx_of_needle)

        # So now we have the first suffix that matches the query
        # We need to then iterate until we find the first suffix that doesn't match the query
        doc_dict = {}
        while True:  # O(n)
            suffix_data = self.__suffixes[idx_of_needle]
            suffix = self.__suffix_of_doc(suffix_data)
            tokens = list(self.__tokenizer.tokens(suffix))

            if normalized_query == tokens[0]:  # i.e. suffix starts with
                if doc_dict[self.__doc_id_from_suffix(suffix_data)]:
                    doc_dict[self.__doc_id_from_suffix(suffix_data)] += 1
                else:
                    doc_dict[self.__doc_id_from_suffix(suffix_data)] = 1
                idx_of_needle += 1
            else:
                break
        results = [{"score": v, "document": self.__corpus[k]}
                   for k, v in doc_dict.items()]
        if "hit_count" in options:
            results = results[:options["hit_count"]]
        return results
