# Content Integration Plan: Office Hours + Talks + Hamel's Evals

## Summary of Sources

### Office Hours (Cohort 3)
- 9 sessions covering weeks 1-5
- Rich practical insights from student questions
- Real-world implementation challenges and solutions
- Business value examples and case studies

### Industry Talks
- 21 talks from practitioners at leading organizations
- Specific performance numbers and benchmarks
- Anti-patterns and mistakes to avoid
- Emerging trends and controversial perspectives

### Hamel's Evals FAQ
- Already partially integrated into Chapter 1
- Additional content available on error analysis methodology
- LLM-as-judge best practices
- Evaluation workflow patterns

## High-Impact Integrations (Priority 1)

### Chapter 1: Starting the Data Flywheel

**Add Precision-Recall Tradeoff Section** (Office Hours week 1-1, 1-2)
- Modern models optimized for recall ("needle in haystack")
- Older models (GPT-3.5) sensitive to low precision
- Testing methodology: different K values, precision-recall curves
- Warning against arbitrary re-ranker thresholds

**Expand Monitoring Section** (Office Hours week 2-2, Talk: Ben & Sidhant)
- Track average cosine distance changes (not absolutes)
- Segment analysis by user cohorts
- Trellis framework for production monitoring
- Implicit vs explicit signals

**Add Multi-turn Conversation Evaluation** (Office Hours week 1-1, 1-2)
- State machine + rubrics hybrid approach
- Extracting criteria scores for logistic regression
- Finding first upstream failure in conversation chains

**Strengthen Anti-patterns Section** (Talk: Skylar Payne)
- 90% of complexity additions perform worse
- 21% silent data loss from encoding issues
- Evaluating only retrieved docs misses false negatives
- Specifics on encoding, staleness, chunking issues

### Chapter 2: From Evaluation to Enhancement

**Expand Hard Negatives Section** (Office Hours week 3-1)
- 30% improvement with hard negatives (vs 6% baseline)
- Concrete methodology for creating hard negatives
- Sources of negative examples from user interactions

**Add Citation Fine-tuning Results** (Office Hours week 5-1)
- 4% → 0% error rate with 1,000 examples
- Validation before fine-tuning critical
- Sample size experimentation

**Expand Re-ranker Section** (Talk: Ayush LanceDB)
- Specific numbers: 12% at top-5, 20% for full-text
- Latency tradeoffs: ~30ms GPU, 4-5x CPU
- Cross-encoder vs bi-encoder explanation

**Add Model Selection Framework** (Office Hours week 3-1)
- BAAI BGE models recommendation
- Systematic testing over "perfect" model search
- Test dimensions: latency, hosting, data volume, performance-cost

### Chapter 3: User Experience and Feedback

**Strengthen Feedback Copy Section** (Talk: Vitor Zapier)
- Specific before/after example with 4x improvement
- "Labeling parties" technique for team alignment
- Growth from 23 → 383 evaluations

**Add Product-as-Sensor Design** (Office Hours week 4-2)
- Building products that "trick" users into labeling
- Examples: chart deletion, citation mouse-overs
- Messaging strategies for feedback collection

**Add Feedback Mining Techniques** (Office Hours week 3-1)
- Citation deletion as negative examples
- Recommendation removal signals
- Email editing before sending

**Expand Implicit Signals** (Talk: Ben & Sidhant)
- User frustration patterns
- Task failures vs completion
- Regeneration frequency

### Chapter 4: Understanding Your Users

**Add Query Clustering Process** (Office Hours week 2-1)
- Summarize → Extract → Embed → Cluster → Label
- Tools: Cura (similar to Claude's Clio)
- Insights extraction methodology

**Add Business Value Framework** (Office Hours week 1-1, week 1-2)
- Inventory vs Capabilities distinction
- Restaurant voice AI: 10% revenue increase example
- Construction contact search: $100K/month problem
- Focus on business outcomes over technical sophistication

**Expand Pricing Models Section** (Office Hours week 4-1)
- Shift from usage-based to outcome-based
- Voice AI: 3% of mechanic's revenue model
- AI as headcount budget vs SaaS budget

### Chapter 5: Building Specialized Capabilities

**Add Tool Portfolio Design** (Office Hours week 5-1, Talk: Beyang Liu)
- Construction example: 4 specialized tools
- Tool naming impacts usage (2% difference)
- Portfolio thinking vs monolithic approach

**Add Document Summarization as Compression** (Office Hours week 5-1)
- Summary designed for specific tasks
- Blueprint example: 16% → 85% recall
- Works for financial reports, multimedia

**Add Temporal Reasoning Section** (Office Hours week 5-1)
- Markdown table format for timestamps
- Two-stage: extract timeline → reason
- Test chronological vs reverse-chronological

**Add Page-Level Chunking** (Office Hours week 2-1)
- Documentation: "which page?" not arbitrary boundaries
- Modern models handle page-sized chunks
- Semantic boundaries respected by authors

**Expand Chunking Section** (Talk: Anton ChromaDB)
- Always examine actual chunks
- Fill context window vs don't group unrelated
- Default settings often far too short
- Semantic vs heuristic approaches

### Chapter 6: Unified Product Architecture

**Add Compute Allocation Strategy** (Office Hours week 3-1)
- Write-time (contextual retrieval) vs read-time (tool use)
- Trade-offs for different use cases
- Medical example: latency constraints favor write-time

**Add Cost Calculation Methodology** (Office Hours week 5-2)
- Calculate token volumes before optimization
- Open source only 8x cheaper example ($60 total)
- Absolute costs vs percentage differences

**Expand Evaluation Data Storage** (Office Hours week 4-1)
- Direct to Postgres vs tracing tool exports
- Schema: session, user, query, chunks, answer
- Build UI on database

## Medium-Impact Integrations (Priority 2)

### Chapter 1
- Data format testing (Markdown vs CSV/JSON) (Office Hours week 5-1)
- Small language models for query rewriting (Office Hours week 1-1)
- Component-based evaluation methodology (Office Hours week 5-1)

### Chapter 2
- Citation source ordering (Office Hours week 5-1)
- Position bias in long contexts
- Metadata extraction as separate ETL jobs (Office Hours week 5-2)

### Chapter 3
- Customer feedback hybrid analysis (Office Hours week 4-2)
- Hierarchical clustering for taxonomy
- Faceted navigation for feedback

### Chapter 5
- Multi-agent vs single-agent trade-offs (Office Hours week 5-1)
- Graph-based RAG skepticism (Office Hours week 2-1)
- Postgres with pgvector (Office Hours week 2-1)

### Chapter 6
- Tool evaluation with plan approval (Office Hours week 5-1)
- Price quote generation process (Office Hours week 5-1)
- Professional styling challenges (Office Hours week 4-2)

## Anti-patterns & Warnings to Add

### Throughout
- Don't cargo cult from Chat LLM era (Talk: Beyang Liu)
- Always examine your data (multiple sources)
- Avoid fully automated evaluation (Talk: Kelly Hong)
- Don't use text embeddings for non-textual data (Talk: Daniel)
- Distinguish adversarial vs merely irrelevant context (Office Hours week 2-2)

## Controversial Perspectives to Consider

These may be too opinionated for the main content but could be valuable:

1. "RAG is dead for coding agents" (Talk: Nik Cline)
2. "Never use evals to guide product development" (Talk: Beyang Liu)
3. "One-shot automation never works" (Talk: Eli Extend)
4. Graph databases often overkill (Office Hours week 2-1)

## Implementation Approach

1. **Phase 1**: Add highest-impact quantitative results and specific techniques
   - Chapter 1: Monitoring, precision-recall, multi-turn eval
   - Chapter 2: Hard negatives (30%), citation fine-tuning, re-ranker numbers
   - Chapter 3: Feedback copy (4x), implicit signals

2. **Phase 2**: Integrate frameworks and methodologies
   - Business value framework
   - Tool portfolio design
   - Compute allocation strategy
   - Query clustering process

3. **Phase 3**: Add anti-patterns and warnings throughout
   - Each chapter gets relevant anti-patterns
   - Consistent "what not to do" sections

4. **Phase 4**: Polish and attribution
   - Add "Further Reading" sections with talk/office hours references
   - Ensure proper attribution for specific numbers/examples
   - Cross-reference between chapters

## Attribution Strategy

- Office Hours insights: "Based on discussions with course participants..."
- Talk insights: "As [Speaker] ([Company]) demonstrated in their presentation..."
- Specific numbers: Always cite source
- Hamel's content: Already has inline attribution in Chapter 1

## Notes

- Focus on production-tested insights over theoretical
- Prioritize specific numbers and concrete examples
- Maintain professional tone (already established)
- Ensure all additions enhance rather than bloat
- Keep chapters focused on core narrative
