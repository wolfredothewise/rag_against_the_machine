UV := uv
UV_CACHE_DIR := /sgoinfre/students/$(USER)/uv_cache
HF_CACHE ?= /sgoinfre/students/$(USER)/hf_cache
VENV_DIR := /sgoinfre/students/$(USER)/rag_venv_milestone

export HF_HOME := $(HF_CACHE)
export HUGGINGFACE_HUB_CACHE := $(HF_CACHE)
export TRANSFORMERS_CACHE := $(HF_CACHE)
export HF_CACHE
export HF_HUB_DISABLE_XET := 1
export CUDA_VISIBLE_DEVICES := ""

.PHONY: install run debug clean lint

install:
	@echo "Preparando el entorno virtual en el disco local de esta maldita máquina..."
	@mkdir -p $(UV_CACHE_DIR)
	@mkdir -p $(VENV_DIR)
	@rm -rf .venv
	@ln -s $(VENV_DIR) .venv
	@echo "Instalando dependencias sin reventar la cuota..."
	@UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync

run:
	@mkdir -p $(HF_CACHE)
	@echo "HF_CACHE=$(HF_CACHE)"

	$(UV) run python -m src.__main__ index --max_chunk_size 2000

	RAG_TARGET=docs RAG_MODE=hybrid $(UV) run python -m src.__main__ search_dataset --dataset_path data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions/docs
	RAG_TARGET=code RAG_MODE=hybrid $(UV) run python -m src.__main__ search_dataset --dataset_path data/datasets_public/public/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions/code
# 	$(UV) run python -m src.__main__ search_single_query --query "How can you limit the number of compilation jobs when building vLLM to avoid system overload?" --k 10
	$(UV) run python -m src.__main__ answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/docs/dataset_docs_public.json --save_directory data/output/answers/AnsweredQuestions/docs
	$(UV) run python -m src.__main__ answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/code/dataset_code_public.json --save_directory data/output/answers/AnsweredQuestions/code
# 	$(UV) run python -m src.__main__ answer "How can you limit the number of compilation jobs when building vLLM to avoid system overload?" --k 10
	$(UV) run python -m src.__main__ evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/docs/dataset_docs_public.json --dataset_path data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json
	$(UV) run python -m src.__main__ evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/code/dataset_code_public.json --dataset_path data/datasets_public/public/AnsweredQuestions/dataset_code_public.json



debug:
	@mkdir -p $(HF_CACHE)
	$(UV) run python -m pdb src/pipeline.py

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -rf *.egg-info
	@rm -rf src/*.egg-info
	@rm -rf src_final/*.egg-info
	@rm -rf .venv
	@rm -rf data/output
	@rm -rf data/processed
	@rm -rf src/__pycache__

lint:
	@echo "Running flake8..."
	@$(UV) run --no-project --with flake8 flake8 src

	@echo "Running mypy..."
	@$(UV) run --no-project --with mypy mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs