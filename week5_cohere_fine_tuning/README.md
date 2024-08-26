The files in this directory create a fine-tuned reranking model on cohere and then test recall on this fine-tuned model.

You can compare it to the [results from week 1](https://github.com/567-labs/systematically-improving-rag/blob/main/week1_bootstrap_evals/metrics.ipynb) where we used a cohere reranker that had not been fine-tuned.

The key files in this directory are:

- `make_synthetic_training_questions.ipynb`: Create the data used to fine-tune our reranker. This is similar in structure to [week1_bootstrap_evals/make_synthetic_questions.ipynb](https://github.com/567-labs/systematically-improving-rag/blob/main/week1_bootstrap_evals/make_synthetic_questions.py)

- `cohere_fine_tuning.ipynb`: Create the fine-tuned model and tests precision/recall
