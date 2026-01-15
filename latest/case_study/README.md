# Systematically Improving RAG: A Complete Case Study

This case study demonstrates the RAG improvement flywheel in action using the WildChat dataset. You'll discover the alignment problem between query generation and embedding strategies, then systematically improve performance from 11% to 82% recall through measured iterations.

**Real-World Connection**: The techniques here mirror the blueprint search case (27% → 85% in 4 days, Workshop Chapter 5.2) and construction company routing (65% → 78%, Chapters 6.1-6.2). Same systematic approach, different contexts.

## Learning Objectives

By completing this case study, you will learn:

- **Workshop Chapter 1 in Practice**: How to systematically evaluate RAG systems with synthetic data
- **Workshop Chapter 2 in Practice**: The critical importance of alignment between queries and embeddings
- **Workshop Chapter 5 in Practice**: Practical techniques for improving retrieval through specialization
- How to measure and compare different embedding strategies (evaluation first!)
- The trade-offs between different approaches to query generation
- When reranking helps and when alignment matters more

**Performance Journey**: v1 queries on first messages (62% recall) → v2 queries on first messages (11% recall) → v5 summaries with v2 queries (55% recall) → understanding the alignment problem is the breakthrough

## Project Structure

```
case_study/
├── README.md                    # This file - your starting point
├── teaching/                    # Step-by-step tutorials
│   ├── part01/                 # Data exploration and statistics
│   ├── part02/                 # The alignment problem
│   ├── part03/                 # Solutions through summaries
│   └── part04/                 # Advanced techniques (coming soon)
├── config.py                   # Configuration settings
├── main.py                     # CLI interface for all operations
├── core/                       # Core implementation
├── data/                       # Generated data and results
│   ├── db.sqlite              # SQLite database with all data
│   ├── results/               # JSON evaluation reports
│   ├── embeddings/            # Vector embeddings
│   └── chromadb/              # ChromaDB vector store
├── pipelines/                  # Data processing pipelines
└── tests/                      # Test suite
```

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (for embedding models)
- ~2GB disk space for data

### Installation

```bash
# Clone the repository
git clone https://github.com/567-labs/systematically-improving-rag.git
cd systematically-improving-rag/latest/case_study

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key
```

### First Run

```bash
# Load 100 conversations for quick testing
uv run python main.py load-wildchat --limit 100

# Check what was loaded
uv run python main.py stats
```

## Database Schema

The case study uses SQLite to store all data with full traceability and rich querying capabilities:

### Core Tables

- **conversation**: Main conversation data (hash, text, language, country, timestamps)
- **question**: Generated questions for evaluation (linked to conversations)
- **summary**: Generated summaries using different techniques
- **evaluation**: Aggregated evaluation metrics (recall@1, recall@5, etc.)
- **evaluationresult**: Detailed evaluation results for every query

### Detailed Results Storage

Unlike many RAG experiments that only save aggregate metrics, this system stores **every query result** with full context:

```sql
-- Example: Get all failed v2 queries with their similarity scores
SELECT er.query, er.rank, er.score, q.version
FROM evaluationresult er
JOIN question q ON er.question_id = q.id
WHERE er.found = 0 AND q.version = 'v2';

-- Example: Compare performance across embedding models
SELECT er.experiment_id, AVG(er.score) as avg_score
FROM evaluationresult er
WHERE er.found = 1
GROUP BY er.experiment_id;
```

### Dual Storage Strategy

- **SQLite**: Structured data for analysis, aggregations, and complex queries
- **JSON**: Complete experiment reports with metadata for reproducibility
- **ChromaDB**: Vector embeddings for similarity search
- **Parquet**: Raw data and embeddings for efficient loading

## Learning Path

### Part 1: Data Exploration and Statistics

**Location**: [`teaching/part01/`](teaching/part01/)

Learn about the WildChat dataset and understand the data you'll be working with:

- Dataset statistics and distribution
- Language and geographic coverage
- Text length analysis
- Data quality considerations

**Key Commands**:

```bash
uv run python main.py load-wildchat --limit 1000
uv run python main.py stats
```

### Part 2: The Alignment Problem

**Location**: [`teaching/part02/`](teaching/part02/)

Discover the fundamental challenge in RAG systems - the alignment between query generation and embedding strategies:

- Understanding v1 vs v2 query strategies
- Why embedding models fail when misaligned
- Experimental results showing 50% performance gaps
- The critical importance of what you embed vs how you search

**Key Commands**:

```bash
uv run python main.py generate-questions --version v1 --limit 1000
uv run python main.py generate-questions --version v2 --limit 1000
uv run python main.py embed-conversations --embedding-model text-embedding-3-small
uv run python main.py evaluate --question-version v1 --embedding-model text-embedding-3-small
uv run python main.py evaluate --question-version v2 --embedding-model text-embedding-3-small
```

### Part 3: Solving the Alignment Problem

**Location**: [`teaching/part03/`](teaching/part03/)

Explore multiple solutions to bridge the alignment gap:

- Full conversation embeddings vs first-message embeddings
- Different summary generation strategies
- Storage and performance trade-offs
- Systematic evaluation of each approach

**Key Commands**:

```bash
uv run python main.py generate-summaries --versions v1,v3,v4 --limit 1000
uv run python main.py embed-summaries --technique v1 --embedding-model text-embedding-3-small
uv run python main.py evaluate --question-version v2 --embedding-model text-embedding-3-small --target-type summary --target-technique v4
```

### Part 4: Advanced Techniques (Coming Soon)

**Location**: [`teaching/part04/`](teaching/part04/)

Advanced optimization techniques including:

- Compare with a sentence transformer cross encoder reranker
- Compare with a cohere reranker

## Key Concepts Covered

### 1. The Alignment Problem

The fundamental insight that you can't search for patterns in embeddings that don't contain pattern information. This is the core challenge in RAG systems.

### 2. Query Generation Strategies

- **v1 (Content-focused)**: Generate queries about specific topics mentioned in conversations
- **v2 (Pattern-focused)**: Generate queries about conversation patterns and dynamics

### 3. Embedding Strategies

- **First-message embeddings**: Fast, small, good for content search
- **Full conversation embeddings**: Large, captures patterns, good for pattern search
- **Summary embeddings**: Balanced approach with multiple variations

### 4. Evaluation Methodology

- Systematic measurement using Recall@1, Recall@5, etc.
- Comparative analysis across different approaches
- Cost-benefit analysis including storage and latency

## Expected Results

By the end of this case study, you'll see results like:

| Approach          | v1 Queries | v2 Queries | Storage | Use Case          |
| ----------------- | ---------- | ---------- | ------- | ----------------- |
| First Message     | 62%        | 12%        | 1x      | Content search    |
| Full Conversation | 55%        | 45%        | 10x     | Pattern search    |
| v1 Summary        | 58%        | 15%        | 2x      | Balanced          |
| v4 Summary        | 52%        | 42%        | 3x      | Pattern-optimized |

## CLI Commands Reference

### Data Loading

```bash
uv run python main.py load-wildchat --limit 1000    # Load conversations
uv run python main.py stats                         # View statistics
```

### Question Generation

```bash
uv run python main.py generate-questions --version v1 --limit 1000
uv run python main.py generate-questions --version v2 --limit 1000
```

### Summary Generation

```bash
uv run python main.py generate-summaries --versions v1,v3,v4 --limit 1000
uv run python main.py generate-summaries --versions all --limit 100
```

### Embedding Creation

```bash
uv run python main.py embed-conversations --embedding-model text-embedding-3-small
uv run python main.py embed-summaries --technique v1 --embedding-model text-embedding-3-small
```

### Evaluation

```bash
uv run python main.py evaluate --question-version v1 --embedding-model text-embedding-3-small
uv run python main.py evaluate --question-version v2 --embedding-model text-embedding-3-small --target-type summary --target-technique v4
```

## Troubleshooting

### Common Issues

1. **OpenAI API Rate Limits**: Use `--limit` for smaller test runs
2. **Memory Issues**: Start with 100 conversations, scale up gradually
3. **ChromaDB Errors**: Ensure all metadata values are non-None
4. **Database Locks**: Only run one operation at a time

### Debug Commands

```bash
# Check database contents
uv run python main.py stats

# Verify embeddings
uv run python verify_embeddings.py

# Check detailed evaluation results
sqlite3 data/db.sqlite "SELECT COUNT(*) FROM evaluationresult;"
sqlite3 data/db.sqlite "SELECT experiment_id, COUNT(*) FROM evaluationresult GROUP BY experiment_id;"

# Query specific results
sqlite3 data/db.sqlite "SELECT query, found, rank, score FROM evaluationresult WHERE experiment_id='your_experiment' LIMIT 5;"

# View detailed logs
tail -f logs/case_study.log
```

## Performance Benchmarks

### Embedding Model Comparison

| Model                                  | Dimensions | v1 Recall@1 | v2 Recall@1 | Cost  |
| -------------------------------------- | ---------- | ----------- | ----------- | ----- |
| sentence-transformers/all-MiniLM-L6-v2 | 384        | 54.8%       | 10.7%       | Free  |
| text-embedding-3-small                 | 1536       | 58.7%       | 11.3%       | $0.02 |
| text-embedding-3-large                 | 3072       | 62.5%       | 12.2%       | $0.13 |

### Processing Times (1000 conversations)

- Question generation: ~5 minutes
- Summary generation: ~15 minutes
- Embedding creation: ~2 minutes
- Evaluation: ~1 minute

## Educational Value

This case study is designed to be:

- **Hands-on**: Every concept is demonstrated with code
- **Systematic**: Controlled experiments with clear hypotheses
- **Practical**: Real-world dataset and production considerations
- **Measurable**: Quantitative evaluation at every step
- **Traceable**: Every query result stored with full context for deep analysis
- **Scalable**: Start small, scale up as you learn

## Additional Resources

- [WildChat Dataset](https://huggingface.co/datasets/allenai/WildChat-1M)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [RAG Best Practices](https://github.com/567-labs/systematically-improving-rag)

## Support

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review the detailed logs in `logs/case_study.log`
3. Ensure your OpenAI API key is properly configured
4. Start with smaller limits (--limit 10) for testing

---

**Ready to start?** Begin with [Part 1: Data Exploration](teaching/part01/) to understand your dataset, then work through each part systematically.

The key insight you'll discover: **In RAG systems, alignment between queries and embeddings matters more than model sophistication.**

---
