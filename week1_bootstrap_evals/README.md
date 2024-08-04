This folder includes code from week 1 of [Systematically Improving RAG Applications`](https://maven.com/applied-llms/rag-playbook).

It shows you how to bootstrap evaluation of a RAG system with synthetic evaluations so you can measure the impact of changes in your retrieval pipelines.

The code uses the example of a hardware e-commerce site building a RAG feature to answer consumer questions based on historical product reviews. The notebook of interest is `make_synthetic_questions.ipynb`. That notebook explains more of what we are trying to achieve with synthetic evaluation.

However a RAG product starts with a corpus of documents to retrieve. For you to be able to run `make_synthetic_questions.ipynb` we also generate synthetic product reviews. That happens in `make_product_reviews.ipynb`. You won't use `make_product_reviews.ipynb` when applying this code to your own projects.

Finally, we calculate metrics on these questions in `metrics.ipynb`. This uses a simple approach to search, but gives you a baseline you can iterate from.
