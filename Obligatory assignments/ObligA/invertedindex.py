#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools
from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable, Iterator, List
from .dictionary import InMemoryDictionary
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .corpus import Corpus
from .posting import Posting
from .postinglist import CompressedInMemoryPostingList, InMemoryPostingList, PostingList


class InvertedIndex(ABC):
    """
    Abstract base class for a simple inverted index.
    """

    def __getitem__(self, term: str) -> Iterator[Posting]:
        return self.get_postings_iterator(term)

    def __contains__(self, term: str) -> bool:
        return self.get_document_frequency(term) > 0

    @abstractmethod
    def get_terms(self, buffer: str) -> Iterator[str]:
        """
        Processes the given text buffer and returns an iterator that yields normalized
        terms as they are indexed. Both query strings and documents need to be
        identically processed.
        """
        pass

    @abstractmethod
    def get_postings_iterator(self, term: str) -> Iterator[Posting]:
        """
        Returns an iterator that can be used to iterate over the term's associated
        posting list. For out-of-vocabulary terms we associate empty posting lists.
        """
        pass

    @abstractmethod
    def get_document_frequency(self, term: str) -> int:
        """
        Returns the number of documents in the indexed corpus that contains the given term.
        """
        pass


class InMemoryInvertedIndex(InvertedIndex):
    """
    A simple in-memory implementation of an inverted index, suitable for small corpora.

    In a serious application we'd have configuration to allow for field-specific NLP,
    scale beyond current memory constraints, have a positional index, and so on.

    If index compression is enabled, only the posting lists are compressed. Dictionary
    compression is currently not supported.
    """

    def __init__(
        self,
        corpus: Corpus,
        fields: Iterable[str],
        normalizer: Normalizer,
        tokenizer: Tokenizer,
        compressed: bool = False,
    ):
        self.__corpus = corpus
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer
        self.__posting_lists: List[PostingList] = []
        self.__dictionary = InMemoryDictionary()
        self.__build_index(fields, compressed)

    def __repr__(self):
        return str({term: self.__posting_lists[term_id] for (term, term_id) in self.__dictionary})

    def __build_index(self, fields: Iterable[str], compressed: bool) -> None:
        token_sequence = []
        for document in self.__corpus:
            for field in fields:
                for term in self.get_terms(document[field]):
                    token_sequence.append(
                        (term,  document.document_id))

        sorted_token_sequence = sorted(token_sequence, key=lambda x: x[0])
        postings = {}
        for term, document_id in sorted_token_sequence:
            if term not in postings:
                postings[term] = []
            postings[term].append(document_id)

        pl = InMemoryPostingList()
        for term, document_ids in postings.items():
            self.__dictionary.add_if_absent(term)
            for document_id, count in Counter(document_ids).items():
                pl.append_posting(Posting(document_id, count))
            self.__posting_lists.append(pl)
            pl = InMemoryPostingList()

    def get_terms(self, buffer: str) -> Iterator[str]:
        return [self.__normalizer.normalize(token) for token in self.__tokenizer.strings(buffer)]

    def get_postings_iterator(self, term: str) -> Iterator[Posting]:
        term_id = self.__dictionary.get_term_id(term)
        if term_id is None:
            return iter([])
        return self.__posting_lists[term_id].get_iterator()

    def get_document_frequency(self, term: str) -> int:
        if self.__dictionary.get_term_id(term) is None:
            return 0
        return self.__posting_lists[self.__dictionary.get_term_id(term)].get_length()
