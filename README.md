# Systematically Improving RAG Applications

A comprehensive educational resource teaching data-driven approaches to building and improving Retrieval-Augmented Generation systems that get better over time. Learn from real case studies with concrete metrics showing how RAG systems improve from 60% to 85%+ accuracy through systematic measurement and iteration.

## What You'll Learn

Transform RAG from a technical implementation into a continuously improving product through:

- **Data-driven evaluation**: Establish metrics before building features
- **Systematic improvement**: Turn evaluation insights into measurable gains
- **User feedback loops**: Design systems that learn from real usage
- **Specialized retrieval**: Build purpose-built retrievers for different content types
- **Intelligent routing**: Orchestrate multiple specialized components
- **Production deployment**: Maintain improvement velocity at scale

### Real Case Studies Featured

**Legal Tech Company**: 63% → 87% accuracy over 3 months through systematic error analysis, better chunking, and validation patterns. Generated 50,000+ citation examples for continuous training.

**Construction Blueprint Search**: 27% → 85% recall in 4 days by using vision models for spatial descriptions. Further improved to 92% for counting queries through bounding box detection.

**Feedback Collection**: 10 → 40 daily submissions (4x improvement) through better UX copy and interactive elements, enabling faster improvement cycles.

### The RAG Flywheel

The core philosophy centers around the "RAG Flywheel" - a continuous improvement cycle that emphasizes:

1. **Measure**: Establish benchmarks and evaluation metrics
2. **Analyze**: Understand failure modes and user patterns
3. **Improve**: Apply targeted optimizations
4. **Iterate**: Continuous refinement based on real-world usage

## Repository Structure

```text
.
├── docs/            # Complete workshop series (Chapters 0-7)
│   ├── workshops/   # Progressive learning path from evaluation to production
│   ├── talks/       # Industry expert presentations with case studies
│   ├── office-hours/# Q&A summaries addressing real implementation challenges
│   └── misc/        # Additional learning resources
├── latest/          # Reference implementations and case study code
│   ├── case_study/  # Comprehensive WildChat project demonstrating concepts
│   ├── week0-6/     # Code examples aligned with workshop chapters
│   └── examples/    # Standalone demonstrations
├── data/            # Real datasets from case studies and talks
└── mkdocs.yml       # Documentation configuration
```

## Learning Path: Workshop Chapters

The workshops follow a systematic progression from evaluation to production:

### Chapter 0: Beyond Implementation to Improvement

Mindset shift from technical project to product. See how the legal tech company went from 63% to 87% accuracy by treating RAG as a recommendation engine with continuous feedback loops.

### Chapter 1: Starting the Data Flywheel

Build evaluation frameworks before you have users. Learn from the blueprint search case: 27% → 85% recall in 4 days through synthetic data and task-specific vision model prompting.

### Chapter 2: From Evaluation to Enhancement

Turn evaluation insights into measurable improvements. Fine-tuning embeddings delivers 6-10% gains. Learn when to use re-rankers vs custom embeddings based on your data distribution.

### Chapter 3: User Experience (3 Parts)

**3.1 - Feedback Collection**: Zapier increased feedback from 10 to 40 submissions/day through better UX copy  
**3.2 - Perceived Performance**: 11% perception improvement equals 40% reduction in perceived wait time  
**3.3 - Quality of Life**: Citations, validation, chain-of-thought delivering 18% accuracy improvements

### Chapter 4: Understanding Users (2 Parts)

**4.1 - Finding Patterns**: Construction company discovered 8% of queries (scheduling) drove 35% of churn  
**4.2 - Prioritization**: Use 2x2 frameworks to choose what to build next based on volume and impact

### Chapter 5: Specialized Retrieval (2 Parts)

**5.1 - Foundations**: Why one-size-fits-all fails. Different queries need different approaches  
**5.2 - Implementation**: Documents, images, tables, SQL - each needs specialized handling

### Chapter 6: Unified Architecture (3 Parts)

**6.1 - Query Routing**: Construction company: 65% → 78% through proper routing (95% × 82% = 78%)  
**6.2 - Tool Interfaces**: Clean APIs enable parallel development. 40 examples/tool = 95% routing accuracy  
**6.3 - Performance Measurement**: Two-level metrics separate routing failures from retrieval failures

### Chapter 7: Production Considerations

Maintain improvement velocity at scale. Construction company: 78% → 84% success while scaling 5x query volume and reducing unit costs from $0.09 to $0.04 per query.

- Part 1: Understanding different content types
- Part 2: Implementation strategies
- **Topics**:
  - Working with documents, images, tables, and structured data
  - Metadata filtering and Text-to-SQL integration
  - PDF parsing and multimodal embeddings

### Week 6: Architecture & Product Integration

## Technologies & Tools

The workshops use industry-standard tools for production RAG systems:

- **LLM APIs**: OpenAI, Anthropic, Cohere
- **Vector Databases**: LanceDB, ChromaDB, Turbopuffer
- **Frameworks**: Sentence-transformers, BERTopic, Transformers, Instructor
- **Evaluation**: Synthetic data generation, precision/recall metrics, A/B testing
- **Monitoring**: Logfire, production observability patterns
- **Processing**: Pandas, SQLModel, Docling for PDF parsing

## Documentation

The `/docs` directory contains comprehensive workshop materials built with MkDocs:

### Content Overview

- **Workshop Chapters (0-7)**: Complete learning path from evaluation to production
- **Office Hours**: Q&A summaries addressing real implementation challenges
- **Industry Talks**: Expert presentations on RAG anti-patterns, embedding performance, production monitoring
- **Case Studies**: Detailed examples with specific metrics and timelines

### Core Philosophy

1. **Product mindset**: RAG as evolving product, not static implementation
2. **Data-driven improvement**: Metrics and feedback guide development
3. **Systematic approach**: Structured improvement processes over ad-hoc tweaking
4. **User-centered design**: Focus on user value, not just technical capabilities
5. **Continuous learning**: Systems that improve with every interaction

Build and view documentation:

```bash
mkdocs serve  # Local development with live reload
mkdocs build  # Static site generation
```

## Getting Started

### Important: Use the `latest/` Directory

**⚠️ Always work in the `latest/` directory for the most current course content.**

The `cohort_1/` and `cohort_2/` directories contain materials from previous course iterations and are kept for reference only. All new development and course work should be done in `latest/`.

### Prerequisites

- Python 3.11 (required - the project uses specific features from this version)
- `uv` package manager (recommended) or `pip`

### Installation

1. Clone the repository
2. Navigate to the `latest/` directory:
   ```bash
   cd latest/
   ```
3. Install dependencies:

   ```bash
   # Using uv (recommended)
   uv install

   # Or using pip
   pip install -e .
   ```

4. Start with `week0/` for the most up-to-date content
5. Follow the notebooks in sequential order within each week
6. Reference the corresponding book chapters in `/docs` for deeper understanding

### Code Quality

Before committing changes, run:

```bash
# Format and fix code issues
uv run ruff check --fix --unsafe-fixes .
uv run ruff format .
```

## Philosophy

This course emphasizes:

- **Systematic Improvement**: Data-driven approaches over guesswork
- **Product Thinking**: Building RAG systems that solve real problems
- **Practical Application**: Real-world datasets and examples
- **Evaluation-First**: Measure before and after every change
- **Continuous Learning**: The field evolves rapidly; the flywheel helps you adapt

## Additional Resources

- Industry talk transcripts in `/data/`
- Office hours recordings summaries in `/docs/office_hours/`
- Advanced notebooks in `/latest/extra_kura/` for clustering and classification topics
- Complete case study implementation in `/latest/case_study/`

## License

This is educational material for the "Systematically Improving RAG Applications" course.
