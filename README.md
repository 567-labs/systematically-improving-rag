# Systematically Improving RAG Applications

A comprehensive course on building and improving Retrieval-Augmented Generation (RAG) systems through systematic evaluation and optimization. This repository contains course materials that supplement the [popular Maven course](https://maven.com/applied-llms/rag-playbook). For additional free materials, visit [improvingrag.com](https://improvingrag.com).

## Course Overview

This course teaches you to move beyond trial-and-error RAG development through a data-driven approach. You'll learn to:

- Build robust evaluation frameworks to measure RAG performance objectively
- Fine-tune embedding models for 15-30% performance improvements
- Understand user query patterns through topic modeling and classification
- Enhance retrieval with structured metadata and SQL integration
- Implement sophisticated tool selection and orchestration

## Why This Course Matters

RAG systems often fail to meet user needs because developers lack systematic approaches to improvement. This course provides:

- **Objective Measurement**: Learn to distinguish real improvements from random variation
- **Targeted Optimization**: Identify exactly where your system fails and why
- **Production-Ready Techniques**: Apply methods proven in real-world applications
- **End-to-End Coverage**: From basic retrieval to complex multi-tool orchestration

## Course Structure

### Week 0: Foundation and Environment Setup

Learn the fundamental tools for the course: Jupyter Notebooks, LanceDB for vector search, and Pydantic Evals for systematic evaluation.

**Key Skills**: Vector databases, hybrid search, evaluation frameworks

### Week 1: RAG Evaluation Foundations

Build a comprehensive evaluation framework using synthetic data generation, retrieval metrics, and statistical validation.

**Key Skills**: Synthetic question generation, recall@k, MRR@k, bootstrapping, statistical significance testing

### Week 2: Embedding Fine-tuning

Fine-tune embedding models using both managed services (Cohere) and open-source approaches (sentence-transformers) for significant performance gains.

**Key Skills**: Hard negative mining, triplet loss training, model deployment

### Week 4: Query Understanding

Apply topic modeling to discover user query patterns and build classification systems for ongoing monitoring.

**Key Skills**: BERTopic, query classification, pattern discovery, satisfaction analysis

### Week 5: Structured Data & Metadata

Enhance RAG with structured metadata filtering, SQL integration, and PDF parsing for handling complex queries.

**Key Skills**: Metadata extraction, hybrid retrieval, Text-to-SQL, document parsing

### Week 6: Tool Selection

Evaluate and improve tool selection in multi-tool RAG systems through systematic testing and prompting strategies.

**Key Skills**: Tool orchestration, precision/recall for tools, few-shot prompting

## Getting Started

### Important Guides

Before starting, please read:

- **[Notebook Versions Guide](latest#readme)** - Explains the different notebook versions (standard, logfire, modal)
- **[Setup Verification](latest/week0/00_setup_and_verification.ipynb)** - Run this first to verify your environment

### Prerequisites

- Python 3.9 (required for BERTopic dependency)
- Basic knowledge of Python, machine learning concepts, and APIs
- API keys for various services (see Environment Setup)

### Installation with uv (Recommended)

First, install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create a virtual environment and install dependencies:

```bash
# Create a virtual environment with Python 3.9
uv venv --python 3.9

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install all dependencies
uv sync
```

### Alternative: Installing with pip

```bash
pip install -e .
```

### Environment Setup

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Required API keys:

- **COHERE_API_KEY**: Production key (not trial) from [Cohere](https://docs.cohere.com/v2/docs/rate-limits)
- **OPENAI_API_KEY**: From [OpenAI](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)
- **HF_TOKEN**: Write-enabled token from [Hugging Face](https://huggingface.co/docs/hub/en/security-tokens)
- **LOGFIRE_TOKEN**: From [Pydantic Logfire](https://logfire.pydantic.dev/docs/how-to-guides/create-write-tokens/)
- **BRAINTRUST_API_KEY**: From [Braintrust](https://www.braintrust.dev/docs/reference/api/ApiKeys)

Load environment variables in notebooks:

```python
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
```

## Key Learning Outcomes

By completing this course, you'll be able to:

1. **Measure What Matters**: Build evaluation frameworks that objectively measure RAG performance
2. **Improve Systematically**: Apply data-driven techniques instead of random experimentation
3. **Handle Complex Queries**: Support queries requiring metadata filtering, SQL access, and multi-tool coordination
4. **Deploy with Confidence**: Verify improvements are statistically significant before production
5. **Scale Effectively**: Apply these techniques to any RAG application or domain

## Course Materials

### Notebooks

Each week contains 2-4 Jupyter notebooks with hands-on exercises. Notebooks include:

- Detailed explanations of concepts
- Working code examples
- Visualization of results
- Best practices and tips

### Datasets

- Bird-Bench Text-to-SQL dataset for evaluation
- Synthetic transaction data for embedding fine-tuning
- Klarna FAQ pages for query understanding
- Clothing dataset for metadata extraction
- 70+ commands for tool selection evaluation

### Office Hours

The `office_hours/` directory contains transcripts and summaries from live sessions, providing additional insights and Q&A content.

## Development Setup

### Pre-commit Hooks

For contributors, install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

These hooks ensure code quality through:

- Black formatting
- Ruff linting with auto-fixes
- YAML validation
- Large file prevention

## Troubleshooting

### Running Outside Jupyter

When running code in Python files instead of notebooks, wrap async calls:

```python
import asyncio
asyncio.run(main())
```

### Visualization Issues

Notebooks include built-in visualizations. If running outside Jupyter, you may need to explicitly call `plt.show()` for matplotlib plots.

## Community and Support

- Report issues at [GitHub Issues](https://github.com/anthropics/claude-code/issues)
- Visit [improvingrag.com](https://improvingrag.com) for additional resources
- Join the Maven course for live instruction and community access

## Acknowledgments

Special thanks to [Dmitry Labazkin](https://github.com/labdmitriy) for frequent feedback and contributions to improve the notebooks.

---

**Note**: This is an advanced course assuming familiarity with LLMs and basic RAG concepts. For beginners, we recommend starting with introductory materials on vector databases and semantic search before diving into systematic improvement techniques.
