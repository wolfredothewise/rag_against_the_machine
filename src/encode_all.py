"""Encoding utilities: dense embeddings and a simple BM25 implementation.

This module exposes `Encoder` which wraps a dense encoder (Sentence
Transformer) and a small `CustomBM25` class for sparse retrieval.
"""

import re
import math
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


_WORD = re.compile(r"[A-Za-z0-9_]+")
_PARTS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


class Encoder:
    """Wrapper for text encoders: dense transformer + BM25 builder.

    Attributes:
        embeddings_matrix: Cached dense embeddings matrix (rows per document).
        model_bm25: Optional CustomBM25 instance after building BM25.
        model_trans: SentenceTransformer model used for dense embeddings.
    """

    def __init__(self) -> None:
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.model_bm25: Optional["CustomBM25"] = None
        self.model_trans: Any = SentenceTransformer(
            "TaylorAI/bge-micro-v2", cache_folder="/goinfre/vhedo-ga/hf_cache"
        )

    def encode_dense(
            self,
            chunks_to_encode: List[str],
            documents: List[str]) -> np.ndarray:
        """Encode documents into a dense embedding matrix.

        If `chunks_to_encode` is empty, returns a zero matrix with the
        expected embedding dimensionality (384).
        """
        if chunks_to_encode:
            embeddings_matrix = self.model_trans.encode(
                chunks_to_encode,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
        else:
            embeddings_matrix = np.zeros((len(documents), 384),
                                         dtype=np.float32)

        self.embeddings_matrix = embeddings_matrix
        return embeddings_matrix

    def encode_bm25(self, documents: List[str]) -> "CustomBM25":
        """Tokenize documents and build a `CustomBM25` model."""
        tokenized_corpus: List[List[str]] = [self.tokenize_bm25(doc)
                                             for doc in documents]
        self.model_bm25 = CustomBM25(tokenized_corpus)
        return self.model_bm25

    def tokenize_bm25(self, text: str) -> List[str]:
        """Lightweight BM25 tokenizer that also splits CamelCase and digits.

        Returns a list of lowercased tokens and additional subword parts.
        """
        tokens: List[str] = []
        for word in _WORD.findall(text):
            tokens.append(word.lower())
            parts = [p.lower() for p in _PARTS.findall(word.replace("_", " "))
                     if len(p) > 1]
            if len(parts) > 1:
                tokens.extend(parts)
        return tokens


class CustomBM25:
    """A minimal BM25 implementation over a tokenized corpus.

    This implementation stores document frequencies, idf scores and the
    per-document term frequencies needed to compute BM25 scores.
    """

    def __init__(self, tokenized_corpus: List[List[str]],
                 k1: float = 1.2, b: float = 0.75) -> None:
        self.k1: float = k1
        self.b: float = b
        self.N: int = len(tokenized_corpus)
        self.avgdl: float = 0.0

        total_length: int = 0
        for doc in tokenized_corpus:
            total_length += len(doc)
        self.avgdl = total_length / self.N if self.N > 0 else 1.0

        self.df: Counter[str] = Counter()
        for doc in tokenized_corpus:
            self.df.update(set(doc))

        self.idf: Dict[str, float] = {}
        for word, freq in self.df.items():
            self.idf[word] = math.log(((self.N - freq + 0.5)
                                       / (freq + 0.5)) + 1.0)

        self.doc_term_freqs: List[Counter[str]] = [Counter(doc) for doc
                                                   in tokenized_corpus]
        self.doc_lens: List[int] = [len(doc) for doc in tokenized_corpus]

    def get_scores(self, query: List[str]) -> np.ndarray:
        """Compute BM25 scores for a tokenized query over all documents.

        Returns an array of shape (N,) with floating point scores.
        """
        scores: np.ndarray = np.zeros(self.N, dtype=np.float32)
        for word in query:
            if word in self.df:
                idf = self.idf[word]
                for i in range(self.N):
                    tf = self.doc_term_freqs[i].get(word, 0)
                    if tf > 0:
                        num = tf * (self.k1 + 1.0)
                        den = tf + self.k1 * (
                            1.0 - self.b + self.b *
                            (self.doc_lens[i] / self.avgdl)
                        )
                        scores[i] += idf * (num / den)
        return scores
