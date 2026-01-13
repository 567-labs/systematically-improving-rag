---
title: The RAG Flywheel
description: Data-Driven Product Development for AI Applications
authors:
  - Jason Liu
date: 2025-04-10
---

# The RAG Flywheel

## A Systematic Approach to Building Self-Improving AI Products

Most RAG implementations struggle in production because teams focus on model selection and prompt engineering while overlooking the fundamentals: measurement, feedback, and systematic improvement.

This guide presents practical frameworks for building RAG systems that become more valuable over time through continuous learning and data-driven optimization.

## The Problem: Why Most RAG Systems Fail

The failure pattern repeats across organizations:

- **Week 1-2:** Demo performs well on prepared examples
- **Week 3-4:** Users report irrelevant results for real queries
- **Week 5-6:** Team debates model alternatives without measurement
- **Week 7-8:** Prompt engineering efforts yield inconsistent improvements
- **Week 9+:** Usage drops as users lose confidence

The issue isn't technology—it's process. Without systematic measurement and improvement mechanisms, RAG systems degrade as user expectations evolve and edge cases accumulate. The legal tech system from the introduction avoided this trap by implementing evaluation from day one, identifying three distinct failure modes, and building specialized solutions for each pattern.

## The Solution: The RAG Improvement Flywheel

### [Introduction: The Product Mindset Shift](workshops/chapter0.md)

Treating RAG as an evolving product rather than a static implementation fundamentally changes how you approach development, measurement, and improvement.

**Key concepts:** The improvement flywheel • Common failure patterns • Product thinking vs implementation thinking

---

### [Chapter 1: Starting the Data Flywheel](workshops/chapter1.md)

Overcome the cold-start problem using synthetic data techniques. Establish evaluation frameworks and begin measuring improvement within days. The consulting firm case study shows how 200 synthetic queries established baselines that led to 40-point recall improvements.

**Topics:** Synthetic evaluation datasets • Precision/recall frameworks • Leading vs lagging metrics • Experiment velocity tracking • Production monitoring with the Trellis framework

---

### [Chapter 2: From Evaluation to Enhancement](workshops/chapter2.md)

Transform evaluation insights into systematic improvements. Just 6,000 examples can yield 6-10% performance gains through embedding fine-tuning. Re-rankers provide 12-20% improvements with proper implementation. Hard negatives are the secret—they drive 30% gains vs 6% baseline improvements.

**Topics:** Embedding fine-tuning with contrastive learning • Re-ranker integration (12% improvement at top-5) • Hard negative mining strategies • Fine-tuning cost realities ($100s, not $1000s)

---

### [Chapter 3: User Experience and Feedback](workshops/chapter3-1.md)

Design interfaces that collect high-quality feedback. Changing "How did we do?" to "Did we answer your question?" increases feedback 5x (0.1% to 0.5%). Zapier's case study shows how better copy and visibility drove feedback from 10 to 40 submissions daily. Product-as-sensor thinking turns every interaction into training data.

**Topics:** High-impact feedback copy patterns • Citation systems for trust building • Implicit signal collection (deletion as negative, selection as positive) • Enterprise Slack integration (5x feedback increase)

---

### [Chapter 4: Understanding Your Users](workshops/chapter4-1.md)

Segment queries to identify high-value patterns. Not all queries deserve equal investment. The 2x2 matrix (volume vs satisfaction) reveals danger zones: high-volume, low-satisfaction segments killing your product. The construction case study shows how 8% of queries (scheduling) drove 35% user churn due to 25% satisfaction.

**Topics:** Query clustering with K-means and the Cura process • 2x2 prioritization matrix • Inventory vs capabilities framework • Business value formula (Impact × Volume % × Success Rate) • User adaptation blindness

---

### [Chapter 5: Building Specialized Capabilities](workshops/chapter5-1.md)

Build purpose-built retrievers for different content types. One-size-fits-all is why most RAG systems underperform. Different queries need different retrievers: exact matching for SKUs, semantic search for concepts, structured queries for attributes. Google didn't stay one search—they built Maps, Images, Scholar, each specialized. The blueprint search case study jumped from 27% to 85% recall by using vision models for spatial descriptions.

**Topics:** Two improvement strategies (metadata extraction vs synthetic text) • RAPTOR for long documents (1,500+ pages) • Tool portfolio design • Two-level measurement (P(correct retriever) × P(correct data | retriever))

---

### [Chapter 6: Unified Product Architecture](workshops/chapter6-1.md)

Integrate specialized components through intelligent routing architectures that direct queries to the right tools while maintaining a simple user experience.

**Topics:** Query routing systems • Tool selection frameworks • Performance monitoring • Continuous improvement pipelines

---

### [Conclusion: Product Principles for AI Applications](misc/what-i-want-you-to-takeaway.md)

Core principles that endure beyond specific models or technologies, providing a foundation for AI product development regardless of how the technology evolves.

## Industry Perspectives and Case Studies

Practitioners from organizations building production RAG systems share their experiences, failures, and insights.

### Selected Talks

**[How Zapier Improved Their AI Feedback Collection](talks/zapier-vitor-evals.md)** - Practical changes that increased feedback volume and quality

**[Re-rankers and Embedding Fine-tuning](talks/fine-tuning-rerankers-embeddings-ayush-lancedb.md)** - When and how to use re-rankers for retrieval improvement

**[When RAG Isn't the Right Solution](talks/rag-is-dead-cline-nik.md)** - Why some coding agents moved away from embedding-based retrieval

**[Common RAG Anti-patterns](talks/rag-antipatterns-skylar-payne.md)** - Mistakes to avoid when building RAG systems

**[Limitations of Public Benchmarks](talks/embedding-performance-generative-evals-kelly-hong.md)** - Why MTEB rankings don't always predict production performance

[View all talks →](talks/index.md)

## Who This Book Is For

**Product Leaders**

- Establish metrics that align with business outcomes
- Build frameworks for prioritizing AI product improvements
- Develop product roadmaps based on data rather than intuition
- Communicate AI capabilities and limitations effectively

**Engineers**

- Implement systems designed for rapid iteration and continuous improvement
- Make architectural decisions that support evolving requirements
- Build modular, specialized capabilities that can be composed and extended
- Manage technical debt in AI systems

**Data Scientists**

- Create synthetic evaluation datasets for cold-start scenarios
- Segment and analyze user queries to identify patterns
- Measure retrieval effectiveness beyond simple accuracy metrics
- Build feedback loops that enable continuous learning

## About the Author

Jason Liu is a machine learning engineer who has worked on computer vision and recommendation systems at Facebook and Stitch Fix. He has helped organizations implement data-driven RAG systems and teaches practical approaches to building AI products that improve over time.
