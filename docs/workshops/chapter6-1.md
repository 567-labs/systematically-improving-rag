---
title: Query Routing Foundations
description: Learn the core principles of building a unified RAG architecture with intelligent query routing
authors:
  - Jason Liu
date: 2025-04-11
tags:
  - query-routing
  - unified-architecture
  - tool-interfaces
---

# Query Routing Foundations: Building a Cohesive RAG System

### Key Insight

**The best retriever is multiple retrievers—success = P(selecting right retriever) × P(retriever finding data).** Query routing isn't about choosing one perfect system. It's about building a portfolio of specialized tools and letting a smart router decide. Start simple with few-shot classification, then evolve to fine-tuned models as you collect routing decisions.

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Understand the query routing problem** - Recognize why even excellent specialized retrievers become useless without proper routing and how to design systems where P(success) = P(right retriever) × P(finding data | right retriever)
2. **Master the tools-as-APIs pattern** - Design clean interfaces between routing logic, tool implementations, and team boundaries that enable parallel development
3. **Organize teams for scalable development** - Structure Interface, Implementation, Router, and Evaluation teams with clear ownership and coordination through well-defined APIs
4. **Design migration strategies** - Move systematically from monolithic to modular RAG systems with clear recognition, separation, interface, and orchestration phases
5. **Apply microservice principles** - Build RAG systems that feel like distributed microservices where specialized services handle specific information retrieval tasks
6. **Implement two-level performance measurement** - Track both routing accuracy and individual retriever performance to identify bottlenecks systematically

These objectives build directly on the specialized retrieval capabilities from Chapter 5 and prepare you for the concrete implementation techniques in [Chapter 6-2](chapter6-2.md).

## What This Chapter Covers

- Building unified RAG architectures with query routing
- Designing tool interfaces for specialized retrievers
- Implementing effective routing between components
- Measuring system-level performance

## Building on Previous Chapters

This is where everything comes together. The journey from Chapter 1 through Chapter 5 built toward this moment:

**The Complete Journey:**

- **Chapter 1**: Established evaluation metrics showing you need 90% routing accuracy × 80% retrieval quality = 72% overall success
- **Chapter 2**: Developed fine-tuning techniques that will improve each specialized retriever once routing directs queries correctly
- **Chapter 3**: Created feedback collection mechanisms that will capture both routing failures and retrieval quality issues
- **Chapter 4**: Performed segmentation analysis revealing the construction company needs three specialized tools: blueprint search (8% of queries, 25% satisfaction), document search (52% of queries, 70% satisfaction), and scheduling (15% of queries, 82% satisfaction)
- **Chapter 5**: Built those specialized retrievers—the blueprint search that jumped from 27% to 85% recall, document processors with contextual retrieval, and structured data tools

**The Missing Piece**: A routing system that directs "Find blueprints with 4 bedrooms" to blueprint search, "What's the safety procedure?" to document search, and "When is the foundation pour?" to schedule lookup.

Without intelligent routing, even the best specialized retrievers sit unused. The construction workers would still get irrelevant results because their queries hit the wrong tools. This chapter shows you how to build the orchestration layer that makes specialization work.

## The Query Routing Problem

In Chapter 5, we built specialized retrievers for different content types. Now we need to decide when to use each one.

**Query routing** means directing user queries to the right retrieval components. Without it, even excellent specialized retrievers become useless if they're never called for the right queries.

### Real-World Example: Construction Company Routing

The construction company from Chapter 4 faced exactly this problem. They had built three excellent specialized retrievers:

- **Blueprint Search**: 85% accuracy when used (up from 27% baseline)
- **Document Search**: 78% accuracy on safety procedures and specifications
- **Schedule Lookup**: 82% accuracy for timeline queries

**The Problem**: With a monolithic system routing all queries to generic search, overall performance was only 65%. Blueprint queries hit document search. Schedule questions went to blueprint search. The specialized tools sat mostly unused.

**The Solution - Week 1**: Implemented basic routing with 10 examples per tool using few-shot classification:

- "Find blueprints with..." → Blueprint Search
- "What's the procedure for..." → Document Search
- "When is the..." → Schedule Lookup

**Results - Week 2**: Routing accuracy reached 88%. Combined with retriever quality:

- Overall success = 88% routing × 80% avg retrieval = 70% (up from 65%)

**The Solution - Week 4**: Added feedback collection tracking which routing decisions led to user satisfaction. Used this data to expand to 40 examples per tool and fine-tune the router.

**Results - Week 6**: Routing accuracy improved to 95%:

- Overall success = 95% routing × 82% avg retrieval = 78%
- User retention improved by 35% (remember from Chapter 4.1?)
- Workers actually started using the system daily

**The Key Formula**: P(success) = P(right tool | query) × P(finding data | right tool)

This decomposition is powerful. When overall performance is 65%, you can't tell if routing is broken (sending queries to wrong tools) or if retrievers are broken (tools can't find answers). Measure both separately to know where to focus improvement efforts.

The architecture we'll build:

1. Uses specialized retrievers built from user segmentation data
2. Routes queries to appropriate components based on learned patterns
3. Provides clear interfaces for both models and users
4. Collects feedback to improve routing accuracy over time

## Compute Allocation: Write-Time vs Read-Time Trade-offs

Before diving into routing mechanics, understand a fundamental design decision: where to invest computational resources. This choice affects system architecture, user experience, and cost structure.

### The Two Approaches

**Write-Time Compute (Contextual Retrieval):**

- Invest processing during indexing/ingestion
- Rewrite chunks to include all necessary context
- Example: Convert "He is unhappy with her" to "Jason the doctor is unhappy with Patient X"
- Makes retrieval simpler and faster
- Anthropic's Claude uses this approach

**Read-Time Compute (Tool Use and Traversal):**

- Store minimal context in chunks
- Use additional compute during retrieval to navigate and connect information
- Similar to how Cursor IDE navigates code (find function → examine context)
- More flexible but can feel slower to users
- Enables dynamic context assembly

### Decision Framework

Choose write-time investment when:

- Data is self-contained (doesn't require external information)
- User wait time is critical (latency-sensitive applications)
- Queries are predictable and well-understood
- Indexing can run offline (overnight jobs, batch processing)

**Medical Records Example**: A healthcare system processes patient records overnight, enriching each chunk with patient demographics, visit dates, and diagnoses. When doctors search during the day, queries return instantly because all context is pre-computed. Latency: 200ms. This approach prioritizes doctor time over processing cost.

Choose read-time investment when:

- Data relationships are complex and dynamic
- Information spans multiple sources or updates frequently
- Users need fresh, up-to-the-moment data
- Storage costs outweigh compute costs

**Legal Research Example**: A law firm's case research system doesn't pre-compute all possible connections between cases, statutes, and precedents. Instead, it dynamically traverses relationships during search based on the specific query. This handles the complexity of 500,000+ cases with millions of relationships without pre-computing every path. Latency: 2-3 seconds. Lawyers accept the wait for comprehensive results.

**The Construction Company's Choice**: They used write-time for blueprint summaries (processed once, queried often) but read-time for schedule lookups (data changes daily). This hybrid approach balanced latency needs with data freshness requirements.

- Context needs vary significantly by query
- Information spans multiple connected documents
- You need flexibility to adjust retrieval strategies
- Example: Code navigation, knowledge graphs, exploratory research

**Medical application example:**

For a medical RAG system with self-contained patient records where minimizing user wait time is critical, write-time investment makes sense. Run overnight jobs to denormalize data—include patient demographics whenever mentioning the patient, add relevant medical history context to each encounter note, pre-compute summary views of longitudinal data.

This approach trades increased storage and preprocessing time for faster, more reliable retrieval that meets strict latency requirements.

**Data normalization parallel:**

This decision mirrors database normalization trade-offs. Do you denormalize data by duplicating phone numbers whenever a person is mentioned (write-time overhead, read-time speed), or keep information normalized and join at query time (write-time simplicity, read-time overhead)?

For RAG systems, the answer depends on your latency budget, data characteristics, and update frequency. 4. Collects feedback to improve routing accuracy

## Tools as APIs Pattern

Treat each specialized retriever as an API that language models can call. This creates separation between:

1. **Tool Interfaces**: Definitions of what each tool does and its parameters
2. **Tool Implementations**: The actual retrieval code
3. **Routing Logic**: Code that selects which tools to call

This is similar to building microservices, except the primary client is a language model rather than another service. The pattern evolved from simple function calling in LLM APIs to more sophisticated tool selection frameworks.

### Benefits of the API Approach

- **Clear Boundaries**: Teams work independently on different tools
- **Testability**: Components can be tested in isolation
- **Reusability**: Tools work for both LLMs and direct API calls
- **Scalability**: Add new capabilities without changing existing code
- **Performance**: Enable parallel execution
- **Team Structure**: Different teams own different components

### Team Organization for Scalable Development

When building these systems at scale, team organization becomes critical. From my experience developing multiple microservices for retrieval at different companies, successful teams organize around these boundaries:

!!! example "Organizational Structure"
**Interface Team** (Product/API Design) - Designs tool specifications based on user research - Defines the contracts between components

- Decides what capabilities to expose - Manages the user experience across tools

  **Implementation Teams** (Engineering)
  - **Search Team**: Builds document and text retrievers
  - **Vision Team**: Handles blueprint and image search
  - **Structured Data Team**: Manages schedule and metadata search
  - Each team optimizes their specific retriever type

  **Router Team** (ML Engineering)
  - Builds and optimizes the query routing system
  - Manages few-shot examples and prompt engineering
  - Handles tool selection accuracy measurement

  **Evaluation Team** (Data Science)
  - Tests end-to-end system performance
  - Identifies bottlenecks between routing and retrieval
  - Runs A/B tests and measures user satisfaction

### Why This Structure Works

This separation allows teams to work independently while maintaining system coherence:

- **Clear ownership**: Each team owns specific metrics and outcomes
- **Parallel development**: Teams can optimize their components simultaneously
- **Scalable expertise**: Teams develop deep knowledge in their domain
- **Clean interfaces**: Teams coordinate through well-defined APIs

**You're effectively becoming a framework developer for language models.** Moving forward, building RAG systems will feel a lot like building distributed microservices, where each service specializes in a particular type of information retrieval.

```mermaid
graph TD
    A[User Query] --> B[Query Router]
    B --> C[Tool Selection]
    C --> D[Document Tool]
    C --> E[Image Tool]
    C --> F[Table Tool]
    D --> G[Ranking]
    E --> G
    F --> G
    G --> H[Context Assembly]
    H --> I[Response Generation]
    I --> J[User Interface]
```

This architecture resembles modern microservice patterns where specialized services handle specific tasks. The difference is that the "client" making API calls is often a language model rather than another service.

### Moving from Monolithic to Modular

Most RAG systems start monolithic: one vector database, one chunking strategy, one retrieval method. This breaks down as content types diversify.

Typical migration path:

1. **Recognition**: Different queries need different retrieval
2. **Separation**: Break into specialized components
3. **Interface**: Define clear contracts between components
4. **Orchestration**: Build routing layer

**Example**: A financial services client migrated from a single vector database to specialized components:

- Development velocity: 40% faster feature delivery
- Retrieval quality: 25-35% improvement by query type
- Team coordination: Fewer cross-team dependencies
- Scaling: New content types added without disrupting existing features

The key was treating each retriever as a service with a clear API contract.

## This Week's Action Items

### System Architecture Planning (Week 1)

1. **Assess Your Current Architecture**
   - [ ] Map your existing RAG system to the monolithic → modular migration phases
   - [ ] Identify which phase you're in: Recognition, Separation, Interface, or Orchestration
   - [ ] Document the specific content types that need different retrieval approaches
   - [ ] Calculate your current system's success rate as P(finding data) baseline

2. **Design Team Organization**
   - [ ] Define roles for Interface, Implementation, Router, and Evaluation teams
   - [ ] Identify which team members have expertise in each specialized domain
   - [ ] Plan coordination mechanisms between teams (APIs, shared evaluation metrics, common tooling)
   - [ ] Establish clear ownership boundaries and success metrics for each team

### Tool Interface Design (Week 1-2)

3. **Implement Tools-as-APIs Pattern**
   - [ ] Design clean API contracts for each specialized retriever from Chapter 5
   - [ ] Separate tool interfaces from implementations to enable parallel development
   - [ ] Create clear parameter specifications that both LLMs and humans can use
   - [ ] Document expected inputs, outputs, and error conditions for each tool

4. **Build Microservice Architecture**
   - [ ] Treat each retriever as an independent service with well-defined boundaries
   - [ ] Design for parallel execution and independent scaling
   - [ ] Implement clear separation between routing logic and retrieval implementations
   - [ ] Plan for testability - each component should be testable in isolation

### Migration Strategy (Week 2-3)

5. **Execute Systematic Migration**
   - [ ] Phase 1 (Recognition): Document query types that need different approaches
   - [ ] Phase 2 (Separation): Break monolithic retriever into specialized components
   - [ ] Phase 3 (Interface): Define clean contracts between all components
   - [ ] Phase 4 (Orchestration): Build routing layer to coordinate specialized tools

6. **Measure Two-Level Performance**
   - [ ] Implement tracking for P(selecting right retriever) - routing accuracy
   - [ ] Implement tracking for P(finding data | right retriever) - individual tool performance
   - [ ] Create dashboards showing both metrics to identify limiting factors
   - [ ] Use performance multiplication to prioritize improvement efforts

### Production Readiness (Week 3-4)

7. **Scale Team Development**
   - [ ] Enable teams to work independently on their specialized components
   - [ ] Implement shared evaluation frameworks across all teams
   - [ ] Create common tooling and standards for interface design
   - [ ] Plan regular coordination meetings focused on API contracts and performance

8. **Prepare for Integration**
   - [ ] Document all tool interfaces in preparation for Chapter 6-2 implementation
   - [ ] Create comprehensive test suites for each specialized component
   - [ ] Plan routing strategies and few-shot example management
   - [ ] Prepare user interface considerations for both AI and direct tool access

### Success Metrics

- **Architecture**: Clear separation of concerns with testable, independent components
- **Team Velocity**: 40% faster feature delivery through parallel development
- **System Performance**: 25-35% improvement in retrieval quality by specialized query type
- **Scalability**: New content types can be added without disrupting existing features
- **Performance Clarity**: Can identify whether bottlenecks are routing or retrieval issues

!!! tip "Next Steps"
In [Chapter 6-2](chapter6-2.md), implement the specific tool interfaces and routing logic that bring this architectural vision to life.
