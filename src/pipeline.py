import glob
import json
import os
import re
from typing import Any

from pydantic import ValidationError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from wordfreq import zipf_frequency

from .chunkers import SuperChunker
from .models import MinimalSource, StudentSearchResults, UnansweredQuestion
from .search_index import HybridRetriever, SearchIndex
from .utils import OverlapEvaluate


class Pipeline:
    """Pipeline for chunking source files and searching questions."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize the pipeline state.

        Args:
            max_chunk_size: Maximum number of characters for each chunk.
        """
        self.max_chunk_size = max_chunk_size
        self.chunker = SuperChunker()
        self.indexator = SearchIndex()
        self.hybrid_rff = HybridRetriever()
        self.list_question: list[UnansweredQuestion] = []
        self.dataset_path = ""

    def start_chunking(self, files_paths: list[str]) -> None:
        """Process each file path and generate chunks for supported files.

        Args:
            files_paths: List of file paths to process.
        """
        for f_path in tqdm(files_paths, desc="Chunking", unit="file"):
            try:
                self.chunker.chunk_orchestator(f_path, self.max_chunk_size)
            except Exception as exc:
                print(f"ERROR while processing {f_path}: {exc}")

    def load_models(self, dataset_path: str, k: int) -> None:
        """Load unanswered questions from a dataset JSON file.

        Args:
            dataset_path: Path to the dataset JSON file.
            k: Number of retrieved candidates to keep. It must be a
                positive integer smaller than 2**31.

        Raises:
            ValueError: If k is not a valid positive integer.
        """
        if not isinstance(k, int) or k <= 0 or k > 2**31 - 1:
            raise ValueError(
                "k must be a positive int and not larger than "
                f"{2**31 - 1}. Received: {k}"
            )
        self.dataset_path = dataset_path

        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                def_content = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Dataset JSON is invalid: {self.dataset_path}"
            ) from exc

        if not isinstance(def_content, dict):
            raise ValueError(
                "Dataset format is invalid: expected a JSON object."
            )

        self.list_question = [
            UnansweredQuestion(**question)
            for question in tqdm(def_content.get(
                "rag_questions", []), desc="Loading questions",
                unit="question")
        ]
        print(f"Loaded {len(self.list_question)} question from the dataset.")

    def search(
        self,
        save_directory: str = "data/processed/",
        k: int | None = None,
    ) -> None:
        """Run hybrid search and export the results to a JSON file.

        Args:
            save_directory: Directory where the output JSON is stored.
            k: Number of sources to retrieve per question.
        """
        if k is None:
            raise ValueError("k must be provided for hybrid search.")

        student_results = self.hybrid_rff.hybrid_algorithm(
            self.list_question, k
        )
        os.makedirs(save_directory, exist_ok=True)
        output_file = os.path.join(
            save_directory,
            os.path.basename(self.dataset_path),
        )
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(student_results.model_dump_json(indent=2))

        print(f"Search results exportaded to: {output_file}")


class CLI:
    """Command-line helpers for the RAG indexing and search flow."""

    def __init__(self) -> None:
        """Initialize the CLI with a pipeline instance."""
        self.pipe = Pipeline()

    def index(self, max_chunk_size: int = 2000) -> None:
        """Index raw project files and ingest chunks into the search index.

        Args:
            max_chunk_size: Maximum size accepted for each generated chunk.
        """
        print("Initializing indexing protocol...")

        try:
            if max_chunk_size <= 0 or max_chunk_size > 2000:
                raise ValueError(
                    "Error max_chunk_size must be between 1 and 2000 "
                    f"and is:{max_chunk_size}"
                )

            self.pipe.max_chunk_size = max_chunk_size
            files_paths = glob.glob(
                "data/raw/vllm-0.10.1/**/*.py", recursive=True
            )
            files_paths += glob.glob(
                "data/raw/vllm-0.10.1/**/*.md", recursive=True
            )

            if not files_paths:
                raise FileNotFoundError(
                    "ERROR: There are not .py or .md files in the selected "
                    "path"
                )

            self.pipe.start_chunking(files_paths)
            self.pipe.indexator.ingest_chunks(self.pipe.chunker.all_chunks)
            print(
                "Ingestion complete! Indices saved under data/processed/. "
                "as .pkl file, for optimization"
            )
        except (
            ValueError,
            FileNotFoundError,
            Exception,
            json.JSONDecodeError,
        ) as e:
            print(e)

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str = "data/processed/",
    ) -> None:
        """Run a batch search over the evaluation dataset.

        Args:
            dataset_path: Path to the dataset JSON file.
            k: Number of retrieved chunks to return for each question.
            save_directory: Directory where results are saved.
        """
        try:
            print(f"Searching data set register: '{dataset_path}' with k={k}")
            name = os.path.basename(dataset_path).lower()
            self.pipe.hybrid_rff.target = "code" if "code" in name else "docs"
            self.pipe.load_models(dataset_path, k)
            self.pipe.search(save_directory, k)
        except FileNotFoundError as exc:
            print(f"ERROR: Dataset file not found: {exc}")
        except ValueError as exc:
            print(f"ERROR: Invalid value for search parameters: {exc}")
        except Exception as exc:
            print(f"ERROR while searching dataset: {exc}")

    def search_single_query(self, query: str, k: int) -> None:
        """Search the index for a single query and print matched sources.

        Args:
            query: User question to search for.
            k: Number of results to retrieve.
        """
        self.pipe.list_question = [
            UnansweredQuestion(question_id="1", question=query)
        ]
        search_result = self.pipe.hybrid_rff.hybrid_algorithm(
            self.pipe.list_question, k
        )
        result = search_result.search_results[0]
        os.makedirs("data/single_output", exist_ok=True)
        output_file = os.path.join(
            "data/single_output",
            os.path.basename("single_search.json"),
        )

        with open(output_file, "w", encoding="utf-8") as f:
            payload = [
                source.model_dump() for source in result.retrieved_sources
            ]
            f.write(json.dumps(payload, indent=2) + "\n")

        print(f"Search results exportaded to: {output_file}")

    def answer(self, query: str, k: int) -> None:
        """Answer a single user question from the previously saved search.

        Args:
            query: User question to answer.
            k: Maximum number of retrieved sources to use as context.
        """
        try:
            if is_gibberish(query):
                raise ValueError("Possible gibberish query")
            with open(
                "data/single_output/single_search.json",
                "r",
                encoding="utf-8",
            ) as f:
                file = json.load(f)
            student_data = [MinimalSource.model_validate(x) for x in file][:k]
            model_name = "Qwen/Qwen3-0.6B"

            cache_dir = (
                os.getenv("HF_HOME")
                or os.getenv("HUGGINGFACE_HUB_CACHE")
                or os.getenv("TRANSFORMERS_CACHE")
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype="auto",
                device_map="auto",
            )
            context = self.pipe.indexator.clean_data.unpack(student_data)
            context_str = "\n\n---\n\n".join(context)
            prompt = (
                "You are an expert technical documentation extractor. "
                "Your ONLY job is to answer the user's question using "
                "the provided context.\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n"
                "Answer:"
            )
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(
                model.device
            )
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
            )
            output_ids = generated_ids[0][
                len(model_inputs.input_ids[0]):
            ].tolist()

            content = tokenizer.decode(output_ids, skip_special_tokens=True)
            print("Answer:", content)

        except FileNotFoundError as exc:
            print("File_path not found", exc)
        except ValidationError as exc:
            print("Validation Error:", exc)
        except json.JSONDecodeError as exc:
            print("JSON structure is bad formed:", exc)
        except Exception as exc:
            print(exc)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ) -> None:
        """Answer every question in a search-results dataset file.

        Args:
            student_search_results_path: Path to a dataset with search results.
            save_directory: Output directory where the serialized answers go.
        """
        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                file = json.load(f)
            student_data = StudentSearchResults.model_validate(file)

            model_name = "Qwen/Qwen3-0.6B"

            cache_dir = (
                os.getenv("HF_HOME")
                or os.getenv("HUGGINGFACE_HUB_CACHE")
                or os.getenv("TRANSFORMERS_CACHE")
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype="auto",
                device_map="auto",
            )

            final_answers: list[dict[str, Any]] = []
            os.makedirs(save_directory, exist_ok=True)
            output_file = os.path.join(
                save_directory,
                os.path.basename(student_search_results_path),
            )

            for data in tqdm(student_data.search_results):
                context = self.pipe.indexator.clean_data.unpack(
                    data.retrieved_sources
                )
                context_str = "\n\n---\n\n".join(context)
                prompt = (
                    "You are an expert technical documentation extractor. "
                    "Your ONLY job is to answer the user's question using "
                    "the provided context.\n"
                    f"Context:\n{context_str}\n\n"
                    f"Question: {data.question}\n"
                    "Answer:"
                )
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(
                    model.device
                )
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                )
                output_ids = generated_ids[0][
                    len(model_inputs.input_ids[0]):
                ].tolist()

                content = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                )

                final_answers.append(
                    {
                        "question_id": data.question_id,
                        "question": data.question,
                        "Answer": content,
                        "sources": [
                            source.model_dump()
                            for source in data.retrieved_sources
                        ],
                    }
                )
                with open(output_file, "w", encoding="utf-8") as o_file:
                    json.dump(
                        final_answers,
                        o_file,
                        indent=4,
                        ensure_ascii=False,
                    )
        except FileNotFoundError as exc:
            print("File_path not found", exc)
        except ValidationError as exc:
            print("Validation Error:", exc)
        except json.JSONDecodeError as exc:
            print(
                "JSON structure is bad formed:",
                exc,
            )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Evaluate retrieval quality for a search results file.

        Args:
            student_search_results_path: Path to the student search results.
            dataset_path: Ground-truth dataset used for evaluation.
        """
        self.overlap_obj = OverlapEvaluate()
        try:
            if "code" in student_search_results_path:
                result = self.overlap_obj.evaluate_acurrancy(
                    student_search_results_path,
                    dataset_path,
                )
                if result >= 0.5:
                    print(f"Recall@5 for code: {result} >= 0.5")
                else:
                    print("Not enough Recall@5", result)

            elif "docs" in student_search_results_path:
                result = self.overlap_obj.evaluate_acurrancy(
                    student_search_results_path,
                    dataset_path,
                )
                if result >= 0.8:
                    print(f"Recall@5 for docs: {result} >= 0.8")
                else:
                    print("Not enough Recall@5", result)

            else:
                print("Unexpected problemoooooo")

        except FileNotFoundError as exc:
            print("File_path not found", exc)
        except json.JSONDecodeError as exc:
            print("JSON structure is bad formed:", exc)


def is_gibberish(text: str, threshold: float = 0.5) -> bool:
    """Return whether the provided text looks like gibberish.

    Args:
        text: Text to classify.
        threshold: Ratio of low-frequency words required to flag gibberish.

    Returns:
        True if the ratio exceeds the threshold; otherwise False.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())

    if not words:
        return True

    unknown = sum(
        1
        for word in words
        if float(zipf_frequency(word, "en")) < 1.0
    )

    return bool(unknown / len(words) > threshold)
