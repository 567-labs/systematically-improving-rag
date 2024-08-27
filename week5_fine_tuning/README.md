The files in this directory create a fine-tuned reranking model on cohere and then test recall on this fine-tuned model.

You can compare it to the [results from week 1](https://github.com/567-labs/systematically-improving-rag/blob/main/week1_bootstrap_evals/metrics.ipynb) where we used a cohere reranker that had not been fine-tuned.

The key files in this directory are:

- `cohere_fine_tuning.ipynb`: Create the fine-tuned model and tests precision/recall

- `sbert.ipynb`: This will test recall both with and without fine-tuning the sentence transformer model. But it does not have fine-tuning yet.
