from .chunkers import SuperChunker
from .search_index import SearchIndex
from .models import UnansweredQuestion, StudentSearchResults, MinimalSource
from .utils import OverlapEvaluate
import glob
import json
import os
from typing import List

from tqdm import tqdm
from .search_index import HybridRetriever
from pydantic import ValidationError

from transformers import AutoModelForCausalLM, AutoTokenizer
import re

from wordfreq import zipf_frequency



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
        self.list_question: List[UnansweredQuestion] = []

    def start_chunking(self, files_paths: List[str]) -> None:
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
            for question in def_content.get("rag_questions", [])
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

        print(
            f"Search results exportaded to: {output_file}"
        )
        


class CLI:
    """Command-line helpers for the RAG indexing and search flow."""

    def __init__(self):
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
                    f"Error max_chunk_size must be between 1 and 2000 "
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
        except (ValueError, FileNotFoundError, Exception, json.JSONDecodeError) as e:
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
            print(
                f"Searching data set register: '{dataset_path}' with k={k}"
            )
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

    def search_single_query(self, query: str, k: int):
        """Search the index for a single query and print matched sources.

        Args:
            query: User question to search for.
            k: Number of results to retrieve.
        """
        # TODO: add extra validation for the provided values.
        self.pipe.list_question = [
            UnansweredQuestion(question_id="1", question=query)
        ]
        search_result = self.pipe.hybrid_rff.hybrid_algorithm(
            self.pipe.list_question, k
        )
        result = search_result.search_results[0]
        os.makedirs("data/output", exist_ok=True)
        output_file = os.path.join("data/output",os.path.basename("single_search.json"))

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps([source.model_dump() for source in result.retrieved_sources],indent=2) + "\n")

        print(f"Search results exportaded to: {output_file}")

    def answer (self, query: str, k: int):
        try:
            if is_gibberish(query):
                raise ValueError("Possible gibberish query")
            with open("data/output/single_search.json", "r", encoding="utf-8") as f: #meterle path a este metodo??
                file = json.load(f)
            student_data = [MinimalSource.model_validate(x) for x in file][:k]
            model_name = "Qwen/Qwen3-0.6B"
            
            cache_dir = "/sgoinfre/students/vhedo-ga/hf_cache"

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
            os.makedirs("data/output/single/answers", exist_ok=True)
            output_file = os.path.join("data/output/single/answers", os.path.basename("single_answer.json"),)
            context = self.pipe.indexator.clean_data.unpack(student_data)
            context_str = "\n\n---\n\n".join(context)
            prompt = (
                "You are an expert technical documentation extractor. "
                "Your ONLY job is to answer the user's question using the provided context. "
                "CRITICAL INSTRUCTION:\n"
                # "Only if the query is none-sense like 'awuqnqls', answer exactly 'Information not found.------------'\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n"
                "Answer:"
                )
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            # conduct text completion
            generated_ids = model.generate( # type:ignore
                **model_inputs,
                max_new_tokens=256
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            content = tokenizer.decode(output_ids, skip_special_tokens=True)
            print("Answer:", content)
            
            
        except FileNotFoundError as e:
            print("File_path not found", e)
        except ValidationError as e:
            print("Validation Error:", e)
        except json.JSONDecodeError as e:
            print("JSON structure is bad formed:", e)
        except Exception as e:
            print(e)

          



    def answer_dataset(self, student_search_results_path: str, save_directory:str):

        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                file = json.load(f)
            student_data = StudentSearchResults.model_validate(file)
        
            model_name = "Qwen/Qwen3-0.6B"

            cache_dir = "/sgoinfre/students/vhedo-ga/hf_cache"

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

            # prepare the model input
            final_answers = []
            os.makedirs(save_directory, exist_ok=True)
            output_file = os.path.join(save_directory, os.path.basename(student_search_results_path),)


            for i, data in tqdm(enumerate(student_data.search_results)):
                context = self.pipe.indexator.clean_data.unpack(data.retrieved_sources)
                context_str = "\n\n---\n\n".join(context)
                prompt = (
                    "You are an expert technical documentation extractor. "
                    "Your ONLY job is to answer the user's question using the provided context. "
                    "CRITICAL INSTRUCTION:\n"
                    # "If the answer is not in the context, output exactly 'Information not found.'\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Question: {data.question}\n"
                    "Answer:"
                    )
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                # conduct text completion
                generated_ids = model.generate( # type:ignore
                    **model_inputs,
                    max_new_tokens=256
                )
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                
                content = tokenizer.decode(output_ids, skip_special_tokens=True)


                final_answers.append({
                    "question_id": data.question_id,
                    "question": data.question,
                    "Answer": content,
                    "sources": [source.model_dump()for source in data.retrieved_sources],})
                with open(output_file, "w", encoding="utf-8") as o_file:
                    json.dump(final_answers, o_file, indent=4, ensure_ascii=False)
                if i  == 3:
                    break


        except FileNotFoundError as e:
            print("File_path not found", e)
        except ValidationError as e:
            print("Validation Error:", e)
        except json.JSONDecodeError as e:
            print("JSON structure is bad formed:", e)

    def evaluate(self, student_search_results_path: str, dataset_path:str):
        self.overlap_obj = OverlapEvaluate()
        try:
            if "code" in student_search_results_path:
                result = self.overlap_obj.evaluate_acurrancy(student_search_results_path, dataset_path)
                if result >= 0.5:
                    print(f"Recall@5 for code: {result} >= 0.5 ")
                else:
                    print("Not enough Recall@5", result)

            elif "docs" in student_search_results_path:
                result = self.overlap_obj.evaluate_acurrancy(student_search_results_path, dataset_path)
                if result >= 0.8:
                    print(f"Recall@5 for docs: {result} >= 0.8 ")
                else:
                    print("Not enough Recall@5", result)

            else:
                print("Unexpected problemoooooo")

        except FileNotFoundError as e:
            print("File_path not found", e)
        except json.JSONDecodeError as e:
            print("JSON structure is bad formed:", e)



def is_gibberish(text: str, threshold: float = 0.5) -> bool:
    words = re.findall(r"[a-zA-Z]+", text.lower())

    if not words:
        return True

    unknown = sum(
        zipf_frequency(word, "en") < 1
        for word in words
    )

    return unknown / len(words) > threshold