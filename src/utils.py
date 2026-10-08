import json
import re
from typing import Any, List, Tuple

from .models import MinimalSource


class CleanData:
    """Prepare and clean file-based text chunks for encoding."""

    def __init__(self) -> None:
        """Initialize the in-memory chunk storage."""
        self.documents: List[str] = []
        self.metadata: List[Tuple[str, int, int]] = []

    def prepare_chunks_for_encode(
        self,
        chunks_list: List[Tuple[str, int, int]],
    ) -> Tuple[List[str], List[str], List[Tuple[str, int, int]]]:
        """Read, filter and prepare chunks for encoding.

        Args:
            chunks_list: List of (file_path, start_index, end_index) tuples.

        Returns:
            A tuple with the encoded chunks, the contextualized documents, and
            the original metadata.
        """
        self.documents = []
        self.metadata = []

        for chunk in chunks_list:
            file_path, fci, lci = chunk
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            raw_text = content[fci:lci].strip()
            if len(raw_text) < 40:
                continue

            contextualized_text = f"File: {file_path}\n{raw_text}"
            self.documents.append(contextualized_text)
            self.metadata.append((file_path, fci, lci))

        chunks_to_encode: List[str] = []
        for i, meta in enumerate(self.metadata):
            file_path = meta[0]
            if file_path.endswith(".md") or file_path.endswith(".markdown"):
                text_to_encode = self._clean_for_dense(self.documents[i])
            else:
                text_to_encode = self.documents[i]
            chunks_to_encode.append(text_to_encode)

        return chunks_to_encode, self.documents, self.metadata

    def _clean_for_dense(self, text: str) -> str:
        """Clean markdown text by removing noisy formatting tokens.

        Args:
            text: Raw markdown text including optional code fences.

        Returns:
            A normalized text string suitable for dense encoding.
        """
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"[#*_`~]+", " ", text)
        return " ".join(text.split())

    def unpack(
        self,
        retrieved_sources: List[MinimalSource] | MinimalSource,
    ) -> List[str]:
        """Read the original source text for each retrieved source.

        Args:
            retrieved_sources: One or many retrieved source records.

        Returns:
            A list containing the exact file slices referenced by the sources.
        """
        if isinstance(retrieved_sources, MinimalSource):
            retrieved_sources = [retrieved_sources]

        context: List[str] = []
        for data in retrieved_sources:
            validated = MinimalSource.model_validate(data)
            with open(
                validated.file_path,
                "r",
                encoding="utf-8",
            ) as file:
                content = file.read()
            context.append(
                content[
                    validated.first_character_index:
                    validated.last_character_index
                ]
            )
        return context


class OverlapEvaluate:
    """Evaluate overlap-based retrieval quality."""

    def __init__(self) -> None:
        """Initialize the overlap evaluator."""

    def _overlap(
        self,
        stu_src: dict[str, Any],
        true_src: dict[str, Any],
    ) -> bool:
        """Return whether the student source overlaps the ground truth.

        Args:
            stu_src: Student source metadata.
            true_src: Ground-truth source metadata.

        Returns:
            True when the IoU exceeds the 5% threshold.
        """
        stu_start = int(stu_src["first_character_index"])
        true_start = int(true_src["first_character_index"])
        stu_end = int(stu_src["last_character_index"])
        true_end = int(true_src["last_character_index"])

        max_start = max(stu_start, true_start)
        min_end = min(stu_end, true_end)
        intersection_length = max(0, min_end - max_start)
        if intersection_length == 0:
            return False

        len_stu = stu_end - stu_start
        len_true = true_end - true_start
        union_length = len_stu + len_true - intersection_length
        if union_length <= 0:
            return False

        iou = intersection_length / union_length
        return bool(iou >= 0.05)

    def evaluate_acurrancy(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 5,
    ) -> float:
        """Compute the recall score for a student retrieval output.

        Args:
            student_search_results_path: Path to the student search results.
            dataset_path: Path to the ground-truth dataset.
            k: Number of retrieved sources to inspect per question.

        Returns:
            Mean recall across the evaluated questions.
        """
        with open(
            student_search_results_path,
            "r",
            encoding="utf-8",
        ) as student_f:
            student_content = json.load(student_f)["search_results"]
        with open(
            dataset_path,
            "r",
            encoding="utf-8",
        ) as compare_f:
            compare_content = json.load(compare_f)["rag_questions"]

        if len(student_content) != len(compare_content):
            return 0.0

        compare_dict = {
            q["question_id"]: q["sources"] for q in compare_content
        }
        total_recalls = 0.0

        for item in student_content:
            if item["question_id"] not in compare_dict:
                continue

            aciertos_pregunta = 0
            top_k_sources = item["retrieved_sources"][:k]
            for true_src in compare_dict[item["question_id"]]:
                for stu_src in top_k_sources:
                    if true_src["file_path"] == stu_src["file_path"]:
                        if self._overlap(stu_src, true_src):
                            aciertos_pregunta += 1
                            break

            recall_pregunta = (
                aciertos_pregunta / len(compare_dict[item["question_id"]])
            )
            total_recalls += recall_pregunta

        recall_final = total_recalls / len(student_content)
        return recall_final
