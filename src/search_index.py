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
    """Build and persist the retrieval index for processed chunks.

    The index combines dense embeddings and a BM25 model so later queries can
    be scored against the same encoded corpus.
    """

    def __init__(self) -> None:
        """Initialize the encoder and index storage.

        This creates the dense encoder and the data structures required to
        store chunk metadata and embeddings before indexing.
        """
        self.encoder = Encoder()
        self.clean_data = CleanData()
        self.embeddings_matrix: Any = None
        self.model_bm25: Any = None

    def ingest_chunks(self, chunks_list: list[tuple[str, int, int]]) -> None:
        """Encode chunk text, build the retrieval index, and save it to disk.

        Args:
            chunks_list: A list of tuples in the form
                ``(file_path, first_character_index, last_character_index)``.
        """
        chunks_to_encode, documents, metadata = (
            self.clean_data.prepare_chunks_for_encode(chunks_list)
        )

        self.embeddings_matrix = self.encoder.encode_dense(
            chunks_to_encode,
            documents,
        )
        self.model_bm25 = self.encoder.encode_bm25(documents)

        os.makedirs("data/processed", exist_ok=True)
        index_data = {
            "model": self.model_bm25,
            "metadata": metadata,
            "documents": documents,
            "embeddings_matrix": self.embeddings_matrix,
        }

        with open("data/processed/hybrid_index.pkl", "wb") as file:
            pickle.dump(index_data, file)

        index_size_kb = os.path.getsize(
            "data/processed/hybrid_index.pkl") / 1024
        print(f" -> Index size: {index_size_kb} KB")


class HybridRetriever:
    """Load a persisted index and rank retrieval results by hybrid scoring.

    The retriever combines BM25 lexical signals with dense semantic signals,
    then filters the final candidates by dataset target type.
    """

    def __init__(self) -> None:
        """Initialize the retriever and the active target mode."""
        self.encoder = Encoder()
        self.target = os.environ.get("RAG_TARGET", "all")
        self.model: Any = None
        self.metadata: list[Any] = []
        self.embeddings_matrix: Any = None

    def load_index(self) -> tuple[Any, list[Any], list[str], Any]:
        """Load the precomputed retrieval index from disk.

        Returns:
            A tuple containing the BM25 model, chunk metadata, documents, and
            the dense embedding matrix.
        """
        index_path = "data/processed/hybrid_index.pkl"
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

    def hybrid_algorithm(self, querys: Any, k: int) -> StudentSearchResults:
        """Retrieve the top sources for each question using hybrid ranking.

        Args:
            querys: Iterable of question objects with ``question_id`` and
                ``question`` attributes.
            k: Maximum number of returned sources per question.

        Returns:
            A ``StudentSearchResults`` object containing the fused retrieval
            results for each query.
        """
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
                scores_bm25,
                scores_dense,
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
        top_k_bm25_indices: list[int],
        top_k_dense_indices: Any,
        k: int,
        k_rrf: int = 60,
    ) -> list[MinimalSource]:
        """Fuse BM25 and dense rankings with reciprocal rank fusion.

        Args:
            top_k_bm25_indices: Candidate indices from the lexical ranker.
            top_k_dense_indices: Candidate indices from the dense ranker.
            k: Maximum number of sources to keep.
            k_rrf: Reciprocal-rank fusion constant used by the RRF formula.

        Returns:
            A ranked list of minimal source records that matches the current
            target type.
        """
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
        retrieved_sources: list[MinimalSource] = []

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
        """Return the BM25 and dense retrieval weights for the active mode.

        Returns:
            A tuple ``(bm25_weight, dense_weight)`` used during hybrid ranking.
        """
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

    def _apply_target_filter(
        self,
        scores_bm25: Any,
        scores_dense: Any,
    ) -> tuple[Any, Any]:
        """Mask scores for chunks that do not match the selected target type.

        Args:
            scores_bm25: BM25 scores for every chunk.
            scores_dense: Dense scores for every chunk.

        Returns:
            The filtered score arrays with non-matching target chunks set to
            negative infinity.
        """
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
