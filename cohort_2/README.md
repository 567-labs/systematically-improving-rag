# Systematically Improving RAG -- Cohort 2 Code

To install the requirements, run `pip install -r requirements.txt`. Make sure that you're using the correct Python version - this course requires Python 3.9 because of our `BERTopic` dependency.

If you are using `uv` to install the requirements, you can run the following command to create a virtual environment with Python 3.9:

```
> uv venv --python 3.9
Using CPython 3.9.6 interpreter at: /Library/Developer/CommandLineTools/usr/bin/python3
Creating virtual environment at: .venv
```

You can then install the dependencies with

```bash
uv pip install -r requirements.txt
```

Our recommendation is to use [uv](https://docs.astral.sh/uv/) where possible

## Checklist

Before starting the course, please make sure that you've done the following

1. Set up your environment variables

> **Important** :
>
> 1. Your Cohere Key must be a PRODUCTION key and not a trial key.
> 2. Make sure that you've bumped yourself to higher tier on Braintrust so that you don't run into rate limit issues.
> 3. Your HF_Token must have write access so that you can upload the models that you'll be training in week 2

We've provided a `.env.example` file in the repository that shows the environment variables that you'll need for this course. You should **copy the contents of the file to a `.env` file at the same path** and fill in the variable before starting to work through the notebooks.

Here are some links on how to get your individual API keys

- [Getting a Cohere API Keys](https://docs.cohere.com/v2/docs/rate-limits)
- [Getting an OpenAI Key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)
- [Getting your Hugging Face Token](https://huggingface.co/docs/hub/en/security-tokens)
- [Getting your Logfire Token](https://logfire.pydantic.dev/docs/how-to-guides/create-write-tokens/)
- [Getting your Logfire Read Token](https://logfire.pydantic.dev/docs/how-to-guides/query-api/)
- [Getting your Braintrust API Key](https://www.braintrust.dev/docs/reference/api/ApiKeys)

You can either set these environment variables in your shell ( [Here is an article on how to do so](https://www3.ntu.edu.sg/home/ehchua/programming/howto/Environment_Variables.html) or run the following code snippet in your notebooks to load it from the `.env` file we've provided.

```python
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
```

This should print out `True` to indicate that it's loaded in the variables successfully.

To verify that the shell variables have been set in the notebook, run the command below.

```
!echo $OPENAI_API_KEY
```

This should print out the value of `OPENAI_API_KEY` that you provided in the `.env` file. You can do the same for the other shell variables and verify they have been set.

## Notebook Overview

Here is an overview of the notebooks by week. Special thanks to [Dmitry Labazkin](https://github.com/labdmitriy) for his frequent feedback and help to make the notebooks better!

**Week 1: RAG Evaluation Foundations**

In Week 1, we'll cover how to systematically evaluate RAG systems with synthetic data. We'll benchmark the performance of different retrieval strategies and then apply simple statistical tests to verify our improvements.

- _Notebook 1:_ Generate synthetic questions to test our retrieval system using a Text-2-SQL example and establish core retrieval metrics.

- _Notebook 2:_ Compare the performance of different retrieval strategies (embedding vs hybrid search) using our synthetic evaluation dataset.

- _Notebook 3:_ Apply statistical validation techniques to verify that our improvements are meaningful and reproducible.

**Week 2: Embedding Fine-tuning**

In Week 2, we'll focus on different approaches to fine-tuning embedding models, starting with managed services and progressing to open-source solutions. We'll explore how even small datasets can yield significant improvements in retrieval performance through careful fine-tuning.

- _Notebook 1:_ Generate synthetic transaction data for fine-tuning using a systematic approach to create diverse, challenging test cases.

- _Notebook 2:_ Fine-tune Cohere's managed re-ranker service, demonstrating how to achieve quick wins with minimal engineering overhead using just a few hundred examples.

- _Notebook 3:_ Implement open-source fine-tuning using sentence-transformers, providing greater control and customization while potentially achieving better performance at lower inference costs.

**Week 4: Query Understanding**

In Week 4, we'll look at how we can understand how users are interacting with our RAG system. We'll do so by leveraging topic modelling, which will help us identify patterns in user queries and prioritise improvements where they matter most. Through a combination of topic modeling and query classification, we'll learn how to identify patterns in user behavior and prioritize improvements where they matter most. Once we've done so, we'll see how we can apply this in production to classify user queries in real-time.

- _Notebook 1:_ Generate a diverse dataset of synthetic user queries based on Klarna's FAQ pages, creating realistic examples that cover different user intents and writing styles.
- _Notebook 2:_ Use BERTopic to automatically discover query patterns and identify problematic areas by combining topic modeling with user satisfaction metrics.
- _Notebook 3:_ Build a flexible classification system that can categorize incoming queries into the patterns we discovered, making it easier to monitor and improve specific areas of our RAG system over time.

**Week 5: Structured Data & Metadata**

In Week 5, we explore how to enhance RAG systems by intelligently handling structured metadata using language models. Starting with a clothing dataset, we demonstrate how to use LLMs to generate consistent metadata tags, build effective filtering mechanisms, and handle complex queries through SQL integration. This progression shows how combining structured data with semantic search leads to more accurate and useful recommendations, particularly for e-commerce applications where users often have specific requirements around attributes like size, color, and price.

- _Notebook 1:_ Use GPT-4o to extract structured metadata from a clothing daaset and build a consistent product catalog that follows a predefined taxonomy.
- _Notebook 2:_ Demonstrates how combining semantic search with metadata filters significantly improves retrieval accuracy, particularly for queries with specific requirements like "Cotton Shirts under $50".
- _Notebook 3:_ How can we integrate SQL queries with retrieval to answer complex questions about product availability, order history and inventory by implementing safe database access through LLMs.
- _Notebook 4:_ : How we can use OCR and Vision Language Models to generate visual citations when building RAG applications dealing with PDFs
- _Notebook 5:_ : How to get started with Cohere's new Embed V4 model to perform embedding search on PDF Document Images

**Week 6: Tool Selection**

In Week 6, we'll explore how to evaluate tool selection in RAG systems. Starting with metrics like precision and recall, we explore how to generate comprehensive test suites to assess tool selection accuracy. Through a combination of system prompts and few-shot examples, we demonstrate techniques to significantly improve a model's ability to select the right tools for user queries.

- _Notebook 1:_ Learn how to measure a model's ability to select appropriate tools through precision and recall metrics and how to choose between parallel and sequential tool selection.
- _Notebook 2:_ Create a comprehensive synthetic dataset that tests different failure modes in tool selection, from handling context-dependent choices to managing multi-step tasks
- _Notebook 3:_ See how we can use system prompts and few-shot examples to improve a model's ability to select the right tools for user queries, measuring the impact of this tool selection against our baseline using the evaluation framework established in earlier notebooks.

Each week builds on previous concepts while introducing new techniques:

- Week 1 establishes evaluation foundations
- Week 2 shows how to improve base retrieval
- Week 4 adds query understanding capabilities
- Week 5 introduces structured data handling
- Week 6 brings everything together in a multi-tool system

The progression moves from basic RAG capabilities to increasingly sophisticated features while maintaining focus on systematic evaluation and improvement.

## Troubleshooting

### Running in a File

If you are running the code in a file instead of a Jupyter Notebook, make sure to wrap the async calls in a asyncio.run method call.

```python
async def main():
    // do something

main()
```

Instead you need to wrap it in an asyncio.run method call as seen below

```python
import asyncio

# This will await the main function and run it
asyncio.run(main())
```

We highly recommend running the code in the Jupyter Notebooks as this will make the code more readable and easier to debug.

Visualisations are also built into the notebooks which makes them much easier to use to understand the data

## Development Setup

### Pre-commit Hooks

This repository uses pre-commit hooks to ensure code quality. To set up pre-commit hooks, run:

```bash
pip install pre-commit
pre-commit install
```

The pre-commit hooks will:

- Run Black for code formatting
- Run Ruff with the --fix option to automatically fix linting issues
- Check for trailing whitespace and file endings
- Validate YAML files
- Check for large files being added

These hooks run automatically when you commit changes, helping maintain consistent code quality across the project.
