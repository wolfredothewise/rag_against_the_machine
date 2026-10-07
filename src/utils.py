from typing import List, Tuple
from .models import MinimalSource
import json

import re


class CleanData:
    """Helper to prepare and clean file-based text chunks for encoding.

    The class reads slices from files, filters out very short chunks,
    and applies a lightweight cleaning step for Markdown documents to
    reduce noisy tokens for dense encoders.

    Attributes:
        documents: Contextualized texts (each prefixed with the file path).
        metadata: Tuples of (file_path, start_index, end_index) matching
            each entry in `documents`.
    """

    def __init__(self) -> None:
        self.documents: List[str] = []
        self.metadata: List[Tuple[str, int, int]] = []

    def prepare_chunks_for_encode(
        self, chunks_list: List[Tuple[str, int, int]]
    ) -> Tuple[List[str], List[str], List[Tuple[str, int, int]]]:
        """Read, filter and prepare chunks for encoding.

        This reads the file slices described by `chunks_list`, ignores very
        short slices, prefixes each slice with the source file path to give
        extra context, and applies markdown cleaning for `.md`/`.markdown`
        files only.

        Args:
            chunks_list: List of (file_path, start_index, end_index) tuples.

        Returns:
            A tuple (chunks_to_encode, documents, metadata):
            - chunks_to_encode: strings ready to be encoded.
            - documents: raw contextualized texts.
            - metadata: original metadata list.
        """
        self.documents = []
        self.metadata = []
        for chunk in chunks_list:
            file_path, fci, lci = chunk
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
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
        """Clean markdown text: remove code blocks, URLs and markup.

        The function returns a single-line normalized string with excess
        whitespace collapsed.

        Args:
            text: Raw markdown text including optional code fences.

        Returns:
            Normalized text suitable for dense encoding.
        """
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"[#*_`~]+", " ", text)
        return " ".join(text.split())

    def unpack(self, retrieved_sources: List[MinimalSource] | MinimalSource)-> List[str]:
        context: List[str] = []
        for data in retrieved_sources:
            data = MinimalSource.model_validate(data)
            with open(data.file_path, "r", encoding="utf-8") as f:
                content = f.read()
                context.append(content[data.first_character_index:data.last_character_index])

            
        return context

class OverlapEvaluate:
    def __init__(self):
        ...
    def _overlap(self, stu_src, true_src) -> bool:
        # 1. Calcular el inicio y el fin de la intersección
        max_start = max(stu_src["first_character_index"], true_src["first_character_index"])
        min_end = min(stu_src["last_character_index"], true_src["last_character_index"])
        intersection_length = max(0, min_end - max_start) # por si hay 0
        if intersection_length == 0:
            return False  # Ni se tocan, a la calle
        # 2. Calcular las longitudes individuales
        len_stu = stu_src["last_character_index"] - stu_src["first_character_index"]
        len_true = true_src["last_character_index"] - true_src["first_character_index"]
        # 3. Calcular la Unión (Suma de ambos menos lo que se solapan para no repetirlo)
        union_length = len_stu + len_true - intersection_length
        if union_length <= 0:
            return False
        # 4. Calcular el IoU y comprobar si pasa el corte del 5% (>= 0.05)
        iou = intersection_length / union_length
        return iou >= 0.05


    def evaluate_acurrancy(self, student_search_results_path,dataset_path, k: int = 5)->float:
        with open(student_search_results_path, "r") as student_f:
            student_content = json.load(student_f)["search_results"]
        with open(dataset_path, "r") as compare_f:
            compare_content = json.load(compare_f)["rag_questions"]

        if len(student_content) != len(compare_content):
            return 0.0
        compare_dict = {q["question_id"]: q["sources"] for q in compare_content}
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
        
            recall_pregunta = aciertos_pregunta / len(compare_dict[item["question_id"]])
            total_recalls += recall_pregunta

            # 6. Calcular la media de todas las preguntas
        recall_final = total_recalls / len(student_content)

        return recall_final
