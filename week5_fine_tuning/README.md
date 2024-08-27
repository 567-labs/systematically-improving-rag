The files in this directory create a fine-tuned reranking model on cohere and then test recall on this fine-tuned model.

You can compare it to the [results from week 1](https://github.com/567-labs/systematically-improving-rag/blob/main/week1_bootstrap_evals/metrics.ipynb) where we used a cohere reranker that had not been fine-tuned.

The key files in this directory are:

- `finetune_sbert.py`: Fine tune a sentence transformer cross-encoder. Run this before `eval_sbert.py`.
- `eval_sbert.py`: Evaluate recall for both the base sentence transformer model and the fine-tuned model
- `cohere_fine_tuning.ipynb`: Create a fine-tuned cohere model and test precision/recall
