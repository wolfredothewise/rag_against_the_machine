"""Encoding utilities: dense embeddings and a simple BM25 implementation.

This module exposes `Encoder` which wraps a dense encoder (Sentence
Transformer) and a small `CustomBM25` class for sparse retrieval.
"""

import os
import re
import math
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


_WORD = re.compile(r"[A-Za-z0-9_]+")
_PARTS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


class Encoder:
    """Wrap the dense embedding model and the BM25 ranker.

    Attributes:
        embeddings_matrix: Cached dense embeddings matrix with one row per
            document.
        model_bm25: Optional BM25 model built from the indexed corpus.
        model_trans: SentenceTransformer instance used for dense encoding.
    """

    def __init__(self) -> None:
        """Initialize the encoder and load the configured model cache."""
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.model_bm25: Optional["CustomBM25"] = None
        cache_dir = (
            os.getenv("HF_HOME")
            or os.getenv("HUGGINGFACE_HUB_CACHE")
            or os.getenv("TRANSFORMERS_CACHE")
            or "/tmp/hf_cache"
        )
        self.model_trans: Any = SentenceTransformer(
            "TaylorAI/bge-micro-v2",
            cache_folder=cache_dir,
        )

    def encode_dense(
            self,
            chunks_to_encode: List[str],
            documents: List[str]) -> np.ndarray:
        """Encode text chunks into a dense embedding matrix.

        Args:
            chunks_to_encode: Text chunks to embed with the dense encoder.
            documents: Full document texts used as a fallback when the chunk
                list is empty.

        Returns:
            A dense embedding matrix with shape ``(n_documents, 384)`` when no
            chunks are provided, or the embeddings for the provided chunks.
        """
        if chunks_to_encode:
            embeddings_matrix = np.asarray(
                self.model_trans.encode(
                    chunks_to_encode,
                    batch_size=32,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
        else:
            embeddings_matrix = np.zeros(
                (len(documents), 384),
                dtype=np.float32,
            )

        self.embeddings_matrix = embeddings_matrix
        return embeddings_matrix

    def encode_bm25(self, documents: List[str]) -> "CustomBM25":
        """Build and cache a BM25 model from a corpus of documents.

        Args:
            documents: Raw document strings to tokenize and index.

        Returns:
            A configured ``CustomBM25`` instance fitted on the provided corpus.
        """
        tokenized_corpus: List[List[str]] = [
            self.tokenize_bm25(doc) for doc in documents
        ]
        self.model_bm25 = CustomBM25(tokenized_corpus)
        return self.model_bm25

    def tokenize_bm25(self, text: str) -> List[str]:
        """Tokenize a text string for BM25 scoring.

        The tokenizer lowercases words and expands CamelCase segments into
        additional sub-tokens for better matching.

        Args:
            text: Raw text to tokenize.

        Returns:
            A list of normalized tokens suitable for BM25 indexing.
        """
        tokens: List[str] = []
        for word in _WORD.findall(text):
            tokens.append(word.lower())
            parts = [
                p.lower()
                for p in _PARTS.findall(word.replace("_", " "))
                if len(p) > 1
            ]
            if len(parts) > 1:
                tokens.extend(parts)
        return tokens


class CustomBM25:
    """Implement a compact BM25 ranking model over a tokenized corpus.

    The class stores document frequencies, inverse document frequencies, and
    per-document term counts needed to compute BM25 scores efficiently.
    """

    def __init__(
        self,
        tokenized_corpus: List[List[str]],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        """Initialize the BM25 scorer.

        Args:
            tokenized_corpus: Tokenized documents used to build the index.
            k1: BM25 saturation parameter.
            b: BM25 length-normalization parameter.
        """
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
            self.idf[word] = math.log(
                ((self.N - freq + 0.5) / (freq + 0.5)) + 1.0
            )

        self.doc_term_freqs: List[Counter[str]] = [
            Counter(doc) for doc in tokenized_corpus
        ]
        self.doc_lens: List[int] = [len(doc) for doc in tokenized_corpus]

    def get_scores(self, query: List[str]) -> np.ndarray:
        """Compute BM25 scores for a tokenized query.

        Args:
            query: Tokens to score against the indexed corpus.

        Returns:
            A NumPy array with one BM25 score per document.
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
                            1.0 - self.b
                            + self.b * (self.doc_lens[i] / self.avgdl)
                        )
                        scores[i] += idf * (num / den)
        return scores
