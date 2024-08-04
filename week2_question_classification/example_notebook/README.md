This folder includes materials from week 2 of [Systematically Improving RAG Applications`](https://maven.com/applied-llms/rag-playbook).

It includes two notebooks. The most important notebook is `analyze_clusters`. This notebook shows how to classify questions from your production app into topics/categories to inform where you focus your attention. You can repurpose this code to run on your own applications.

Because you'll want data when first exploring this code, the notebook uses synthetic data that was generated in `make_prod_questions.ipynb`. This notebook will not be used in your production system. Instead, it is included only to show how we created some test data for the sake of example.

`question_types.py` includes the some interface types (much of which is used in both notebooks.)
