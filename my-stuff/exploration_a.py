
from context import in3120

normalizer = in3120.SimpleNormalizer()
tokenizer = in3120.SimpleTokenizer()
corpus = in3120.InMemoryCorpus()
corpus.add_document(in3120.InMemoryDocument(0, {"body": "this is a Test"}))
corpus.add_document(in3120.InMemoryDocument(1, {"body": "test TEST prØve"}))
index = in3120.InMemoryInvertedIndex(
    corpus, ["body"], normalizer, tokenizer, False)

print(list(index.get_terms("PRøvE wtf tesT")))
print(list(index["prøve"]))
print([(p.document_id, p.term_frequency)
       for p in index["prøve"]])
print([(p.document_id, p.term_frequency)
      for p in index["test"]])  # [(0, 1), (1, 2)]
