"""Okapi BM25 in plain Python: an inverted index and one scoring formula.

score(q, d) = sum over query words w of
    idf(w) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))

tf is how often w occurs in d. idf(w) = ln(1 + (N - df + 0.5) / (df + 0.5)),
where N is the number of documents and df the number that contain w. This idf
is never negative. bm25s calls this idf variant "lucene"; rank_bm25 uses a
different one (see the README).
"""

import heapq
import math
import re
from collections import Counter

WORD = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase words and numbers. No stemming, no stopword list."""
    return WORD.findall(text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        """docs: a list of token lists, one per document."""
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / self.N
        # Inverted index: word -> list of (doc number, tf). Only documents
        # that contain the word are listed, so a query touches few documents.
        self.postings = {}
        for i, doc in enumerate(docs):
            for word, tf in Counter(doc).items():
                self.postings.setdefault(word, []).append((i, tf))

    def idf(self, word):
        df = len(self.postings.get(word, ()))
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def term_score(self, tf, dl, idf):
        norm = 1 - self.b + self.b * dl / self.avgdl
        return idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)

    def scores(self, query):
        """Scores for every document that shares a word with the query.

        A word repeated in the query counts once per repeat, as in rank_bm25
        and bm25s. Documents that share no word are left out (score 0).
        """
        out = {}
        for word in query:
            if word not in self.postings:
                continue
            idf = self.idf(word)
            for i, tf in self.postings[word]:
                out[i] = out.get(i, 0.0) + self.term_score(tf, self.doc_len[i], idf)
        return out

    def score(self, query, i):
        return self.scores(query).get(i, 0.0)

    def search(self, query, k=10):
        """Top k as (doc number, score), highest first. Ties: lower number first."""
        found = self.scores(query)
        top = heapq.nsmallest(k, ((-s, i) for i, s in found.items()))
        return [(i, -neg) for neg, i in top]
