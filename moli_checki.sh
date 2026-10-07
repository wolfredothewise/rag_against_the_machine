./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/docs/dataset_docs_public.json \
  data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
  --k 10 \
  --max_context_length 2000 & ./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/code/dataset_code_public.json \
  data/datasets_public/public/AnsweredQuestions/dataset_code_public.json \
  --k 10 \
  --max_context_length 2000


# ./moulinette evaluate_student_search_results \
#   data/output/search_results/UnansweredQuestions/docs/dataset_docs_private.json \
#   data/datasets/private/AnsweredQuestions/dataset_docs_private.json \
#   --k 10 \
#   --max_context_length 2000 & ./moulinette evaluate_student_search_results \
#   data/output/search_results/UnansweredQuestions/code/dataset_code_private.json \
#   data/datasets/private/AnsweredQuestions/dataset_code_private.json \
#   --k 10 \
#   --max_context_length 2000