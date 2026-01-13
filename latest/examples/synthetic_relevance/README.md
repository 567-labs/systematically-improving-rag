# Synthetic Relevance Evaluation with LLM Judges

This example demonstrates how to use Large Language Models (LLMs) as judges for relevance scoring in search and retrieval systems, with human validation to measure alignment.

## Overview

Many search and RAG systems struggle with relevance evaluation. This tool shows how to:

1. **Use LLMs as judges** to automatically score document relevance
2. **Collect human annotations** for the same documents
3. **Compare alignment** between LLM and human judgments
4. **Calculate meaningful metrics** to assess judge quality

## Why Binary Scoring (0/1)?

We use binary relevance (relevant/not relevant) instead of 5-star scales because:

- ✅ **Simpler decisions**: Clear yes/no choices reduce annotation uncertainty
- ✅ **Higher agreement**: Binary scales show better inter-annotator reliability
- ✅ **Easier analysis**: Simpler to calculate precision, recall, and agreement
- ✅ **Good starting point**: You can always expand to multi-point scales later

## Quick Start

### 1. Setup

```bash
# Install dependencies
uv add openai instructor pydantic "typer[all]" rich

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Recommended Workflow (3-Phase Async)

```bash
# Phase 1: Generate LLM scores for all 40 query-document pairs (fast, async)
uv run python main.py generate-llm-scores

# Phase 2: Add human labels (interactive, resumable)
uv run python main.py label

# Phase 3: Analyze correlation and confidence calibration
uv run python main.py analyze
```

### 3. Alternative: Single Query Demo

```bash
# See what the tool does
uv run python main.py demo

# Run legacy single-query evaluation
uv run python main.py evaluate --query "machine learning algorithms"
```

## What Happens During Evaluation

### Phase 1: Async LLM Generation (`generate-llm-scores`)
The tool processes **4 diverse questions** with **10 documents each** (40 total evaluations):

**Questions:**
- "What are the most effective machine learning algorithms for classification tasks?"
- "What are some healthy breakfast recipes that are quick to prepare?"
- "How is climate change affecting global weather patterns and ecosystems?"
- "What are the essential software engineering best practices for large codebases?"

**For each question-document pair, GPT-4 provides:**
- **Binary relevance score** (0 or 1)
- **Detailed reasoning** explaining the decision
- **Confidence level** (0.0 to 1.0)

All LLM judgments are processed concurrently using `asyncio.gather()` and saved to `llm_scores.json`.

**Immediate Analysis:**
After generation, you'll see detailed performance metrics:
- **Per-question precision/recall**: How well LLM performs on each topic
- **NDCG@10 scores**: Ranking quality (0.0-1.0, higher is better)
- **Overall metrics**: Precision, Recall, F1, and confusion matrix
- **Ground truth comparison**: Uses document metadata (high/medium/low/none relevance)

### Phase 2: Human Annotation (`label`)
Interactive interface that shows:
```
Question: What are the most effective machine learning algorithms for classification tasks?

Document: Support Vector Machines (SVMs) are powerful supervised learning algorithms...

🤖 LLM Judge: ✅ Relevant (confidence: 0.95)
💭 LLM Reasoning: This document directly describes a machine learning algorithm...

Do YOU think this document is relevant to the question? [y/N]:
```

**Features:**
- Progress tracking (e.g., "15/40 completed")
- Resumable (saves progress to `human_annotations.json`)
- Shows LLM judgment before asking for human input

### Phase 3: Correlation Analysis (`analyze`)
Comprehensive analysis including:
- **Agreement rate**: How often LLM and human agree
- **Precision/Recall**: LLM performance metrics
- **Confidence calibration**: Are high-confidence predictions more accurate?
- **Per-query breakdown**: Performance varies by topic
- **Confidence correlation**: Average confidence when correct vs incorrect

## Sample Output

### LLM Performance Analysis (Phase 1)
```
📋 Per-Question Analysis
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┓
┃ Question                             ┃ Rel… ┃ LLM  ┃      ┃      ┃       ┃
┃                                      ┃ Docs ┃ Fou… ┃ Pre… ┃ Rec… ┃ NDCG… ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━┩
│ What are the most effective machine… │ 7    │ 4    │ 100… │ 57.… │ 0.809 │
│ What are some healthy breakfast rec… │ 7    │ 5    │ 100… │ 71.… │ 0.890 │
│ How is climate change affecting glo… │ 7    │ 5    │ 100… │ 71.… │ 0.890 │
│ What are the essential software eng… │ 7    │ 5    │ 100… │ 71.… │ 0.890 │
└──────────────────────────────────────┴──────┴──────┴──────┴──────┴───────┘

📊 Overall Performance
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric              ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Overall Precision   │ 100.0% │
│ Overall Recall      │ 67.9%  │
│ F1 Score            │ 80.9%  │
│ Average NDCG@10     │ 0.870  │
│ Total Relevant Docs │ 28     │
│ LLM Found Relevant  │ 19     │
└─────────────────────┴────────┘
```

### Human vs LLM Analysis (Phase 3)
```
📊 Overall Results
┌───────────────────┬─────────┐
│ Metric            │ Value   │
├───────────────────┼─────────┤
│ Total Evaluations │ 40      │
│ Agreement Rate    │ 82.5%   │
│ LLM Precision     │ 85.7%   │
│ LLM Recall        │ 78.9%   │
└───────────────────┴─────────┘

🎯 Confidence Analysis
┌──────────────────────────┬───────┐
│ Metric                   │ Value │
├──────────────────────────┼───────┤
│ Avg Confidence (Correct) │ 0.891 │
│ Avg Confidence (Incorrect) │ 0.723 │
│ High Confidence Accuracy │ 89.5% │
│ Low Confidence Accuracy  │ 66.7% │
│ Calibration Gap          │ +22.8%│
└──────────────────────────┴───────┘

💡 Key Insights:
✅ LLM shows good confidence calibration - more confident when correct
🎯 High-confidence predictions are significantly more accurate
🎉 High agreement! LLM judge aligns well with human judgment
```

## Key Components

### Models (`models.py`)
- `RelevanceScore`: LLM judgment with reasoning
- `SearchResult`: Document representation
- `RelevanceEvaluation`: Single query-document evaluation
- `EvaluationResults`: Aggregated metrics

### Main Application (`main.py`)
- `mock_search()`: Hardcoded search results
- `get_llm_relevance_score()`: LLM judge using instructor
- `get_human_relevance_score()`: CLI human annotation
- `calculate_metrics()`: Agreement and performance metrics

## Understanding the Results

### High Agreement (80%+)
- LLM judge is well-calibrated
- Prompts are clear and effective
- Ready for production use

### Moderate Agreement (60-80%)
- LLM shows promise but needs improvement
- Consider prompt engineering or fine-tuning
- May work for some use cases

### Low Agreement (<60%)
- Significant alignment issues
- Review prompts and examples
- Consider different models or approaches

## Extending This Example

### 1. Real Search Integration
Replace `mock_search()` with your actual search system:

```python
def real_search(query: str) -> List[SearchResult]:
    # Connect to your search backend
    results = your_search_system.search(query)
    return [SearchResult(id=r.id, content=r.text) for r in results]
```

### 2. Different Models
Try different LLM judges:

```python
# In get_llm_relevance_score()
response = await client.chat.completions.create(
    model="gpt-4",  # or "claude-3-sonnet", etc.
    messages=[{"role": "user", "content": prompt}],
    response_model=RelevanceScore
)
```

### 3. Domain-Specific Prompts
Customize the relevance prompt for your domain:

```python
prompt = f"""
You are an expert in medical literature search.
A user is searching for: {query}

Document: {document.content}

Is this document medically relevant to the query?
Consider clinical relevance, treatment implications, and diagnostic value.
"""
```

### 4. Multi-Point Scales
Extend to 5-point relevance scales:

```python
class RelevanceScore(BaseModel):
    relevance_score: int = Field(ge=1, le=5, description="1=not relevant, 5=highly relevant")
    reasoning: str
```

## Use Cases

### Evaluation Frameworks
- Benchmark different search algorithms
- Test relevance improvements over time
- Compare human vs. automated judgments

### Training Data Generation
- Create relevance training datasets
- Generate synthetic query-document pairs
- Bootstrap cold-start evaluation systems

### Production Monitoring
- Continuously monitor search quality
- Detect relevance drift over time
- Alert on significant judgment disagreements

## Tips for Success

1. **Start Simple**: Binary scoring before multi-point scales
2. **Good Examples**: Include diverse relevant/irrelevant cases
3. **Clear Prompts**: Specific criteria for relevance decisions
4. **Multiple Judges**: Compare multiple LLMs or human annotators
5. **Regular Updates**: Retrain or adjust prompts based on results

## Why This Matters for RAG

Relevance evaluation is crucial for RAG systems because:

- **Quality Assurance**: Ensure retrieval finds truly useful documents
- **Systematic Improvement**: Measure before/after changes
- **Cost Optimization**: Focus expensive LLM calls on relevant content
- **User Experience**: Better relevance means better answers

This example provides a foundation for building robust evaluation frameworks that help you systematically improve your search and RAG systems.

## Next Steps

1. Run the demo to understand the workflow
2. Adapt the search function to your data
3. Customize prompts for your domain
4. Scale up annotation with multiple judges
5. Integrate into your evaluation pipeline

Remember: The goal isn't perfect agreement between LLM and human judges, but rather understanding where they align and disagree so you can make informed decisions about your system's relevance thresholds and improvements.