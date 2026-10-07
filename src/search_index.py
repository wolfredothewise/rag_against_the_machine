"""Hybrid retrieval index and ranker for code and documentation search."""

import os
import pickle
from collections import defaultdict
from typing import Any

import numpy as np

from .encode_all import Encoder
from .models import MinimalSearchResults, MinimalSource, StudentSearchResults
from .utils import CleanData

MODE = os.environ.get("RAG_MODE", "hybrid")
TARGET_DATASET = os.environ.get("RAG_TARGET", "all")


class SearchIndex:
    """Build and persist a hybrid index from processed chunks."""

    def __init__(self) -> None:
        """Initialize the encoder used for dense and BM25 indexing."""
        self.encoder = Encoder()
        self.clean_data = CleanData()
        self.embeddings_matrix = None
        self.model_bm25 = None

    def ingest_chunks(self, chunks_list):
        """Encode input chunks, build the index, and persist it to disk."""
        chunks_to_encode, documents, metadata = (
            self.clean_data.prepare_chunks_for_encode(chunks_list)
        )

        self.embeddings_matrix = self.encoder.encode_dense(
            chunks_to_encode, documents
        )
        self.model_bm25 = self.encoder.encode_bm25(documents)

        os.makedirs("data/processed", exist_ok=True)
        index_data = {
            "model": self.model_bm25,
            "metadata": metadata,
            "documents": documents,
            "embeddings_matrix": self.embeddings_matrix,
        }

        with open("data/processed/bm25_index.pkl", "wb") as file:
            pickle.dump(index_data, file)

        index_size_kb = os.path.getsize("data/processed/bm25_index.pkl") / 1024
        print(f" -> Index size: {index_size_kb} KB")


class HybridRetriever:
    """Load persisted vectors and combine BM25 and dense retrieval."""

    def __init__(self) -> None:
        """Initialize the retriever and active target."""
        self.encoder = Encoder()
        self.target = os.environ.get("RAG_TARGET", "all")
        self.model: Any = None
        self.metadata: list[Any] = []
        self.embeddings_matrix: Any = None

    def load_index(self):
        """Load the precomputed retrieval index from disk."""
        index_path = "data/processed/bm25_index.pkl"
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                "ERROR: El índice no existe. Ejecuta primero el comando "
                "'index'."
            )

        with open(index_path, "rb") as file:
            data = pickle.load(file)
            model = data["model"]
            metadata = data["metadata"]
            documents = data["documents"]
            embeddings_matrix = data["embeddings_matrix"]

        return model, metadata, documents, embeddings_matrix

    def hybrid_algorithm(self, querys, k) -> StudentSearchResults:
        """Retrieve the top sources for each question using hybrid ranking."""
        (
            self.model,
            self.metadata,
            _,
            self.embeddings_matrix,
        ) = self.load_index()
        results_rrf = []

        for query in querys:
            tokenized_query_bm25 = self.encoder.tokenize_bm25(query.question)
            tokenized_query_dense = self.encoder.model_trans.encode(
                query.question,
                normalize_embeddings=True,
            )

            scores_bm25 = self.model.get_scores(tokenized_query_bm25)
            scores_dense = self.embeddings_matrix @ tokenized_query_dense

            scores_bm25, scores_dense = self._apply_target_filter(
                scores_bm25, scores_dense
            )

            valid_bm25 = [
                (idx, score)
                for idx, score in enumerate(scores_bm25)
                if score > 0
            ]
            top_k_bm25_indices = [
                idx
                for idx, _ in sorted(
                    valid_bm25,
                    key=lambda item: item[1],
                    reverse=True,
                )[:200]
            ]

            top_k_dense_indices = np.argsort(-scores_dense)[:200]

            retrieved_sources = self.reciprocal_rank_fusion(
                top_k_bm25_indices,
                top_k_dense_indices,
                k,
            )

            student_research = MinimalSearchResults(
                question_id=query.question_id,
                question=query.question,
                retrieved_sources=retrieved_sources,
            )
            results_rrf.append(student_research)

        return StudentSearchResults(search_results=results_rrf, k=k)

    def reciprocal_rank_fusion(
        self,
        top_k_bm25_indices,
        top_k_dense_indices,
        k: int,
        k_rrf: int = 60,
    ):
        """Fuse BM25 and dense rankings using reciprocal rank fusion."""
        bm25_weight, dense_weight = self._get_search_weights()

        rrf_scores: dict[Any, float] = defaultdict(float)
        if bm25_weight > 0:
            for rank, doc_idx in enumerate(top_k_bm25_indices):
                doc_idx = int(doc_idx)
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + (
                    bm25_weight / (k_rrf + rank + 1)
                )

        if dense_weight > 0:
            for rank, doc_idx in enumerate(top_k_dense_indices):
                doc_idx = int(doc_idx)
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + (
                    dense_weight / (k_rrf + rank + 1)
                )

        ranking_final = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
        retrieved_sources = []

        for index, _ in ranking_final:
            file_path, fci, lci = self.metadata[index]

            if self.target == "docs" and file_path.endswith(".py"):
                continue
            if self.target == "code" and (
                file_path.endswith(".md") or file_path.endswith(".markdown")
            ):
                continue

            source = MinimalSource(
                file_path=file_path,
                first_character_index=int(fci),
                last_character_index=int(lci),
            )
            retrieved_sources.append(source)

            if len(retrieved_sources) == k:
                break

        return retrieved_sources

    def _get_search_weights(self) -> tuple[float, float]:
        """Return the BM25 and dense weights for the active mode."""
        if MODE == "bm25":
            bm25_weight = 1.0
            dense_weight = 0.0
        elif MODE == "dense":
            bm25_weight = 0.0
            dense_weight = 1.0
        else:
            bm25_weight = 1.9
            dense_weight = 1.0

        return bm25_weight, dense_weight

    def _apply_target_filter(self, scores_bm25, scores_dense):
        """Remove chunks that do not match the selected target type."""
        for i, meta in enumerate(self.metadata):
            file_path = meta[0]
            if self.target == "docs" and file_path.endswith(".py"):
                scores_bm25[i] = float("-inf")
                scores_dense[i] = float("-inf")
            elif self.target == "code" and (
                file_path.endswith(".md") or file_path.endswith(".markdown")
            ):
                scores_bm25[i] = float("-inf")
                scores_dense[i] = float("-inf")

        return scores_bm25, scores_dense
