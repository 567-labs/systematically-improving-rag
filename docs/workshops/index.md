---
title: Workshops
description: Hands-on workshops for building self-improving RAG systems
---

# Workshops

These workshops walk you through building RAG systems that get better over time through systematic measurement and improvement.

## What's Covered

### [Introduction: Beyond Implementation to Improvement](chapter0.md)

Why most RAG systems fail after deployment and how to build ones that improve instead. See how a legal tech company went from 63% to 87% accuracy over three months by treating RAG as a recommendation engine with continuous feedback loops. Learn to distinguish inventory problems from capability problems and move from random tweaks to data-driven improvements.

### [Chapter 1: Getting Started with Synthetic Data](chapter1.md)

How to evaluate your RAG system before you have real users. Learn to avoid common mistakes (vague metrics, generic solutions), generate synthetic evaluation data, and set up continuous evaluation pipelines. Real examples: blueprint search improving from 27% to 85% recall in four days, consultant interview system jumping from 63% to 72% accuracy through error analysis.

### [Chapter 2: From Evaluation to Better Models](chapter2.md)

Turn your evaluation data into actual improvements. Covers when generic embeddings fail (asymmetric queries), how to create training data from evaluations, fine-tuning strategies that deliver 6-10% improvements, and cost-effective alternatives like re-rankers. Learn the four-step loop: evaluate, generate training data, fine-tune, and measure impact.

### Chapter 3: Getting Users to Actually Give Feedback

#### [Chapter 3.1: Feedback Collection That Works](chapter3-1.md)

How to get feedback rates above 30% (most systems get less than 1%). See how Zapier increased feedback submissions from 10 to 40 per day through better copy and UI design. Includes specific copy that works, UI patterns, mining implicit signals, and Slack integration examples that achieve 50,000+ examples collected.

#### [Chapter 3.2: Making RAG Feel Fast](chapter3-2.md)

Streaming techniques that make your system feel faster and increase feedback by 30-40%. Learn why perceived speed matters more than actual speed (11% perception improvement equals 40% reduction in perceived wait time). Covers Server-Sent Events, skeleton screens, and platform-specific tricks for Slack and web.

#### [Chapter 3.3: Small Changes, Big Impact](chapter3-3.md)

Practical improvements that users love: interactive citations (generating 50,000+ examples for training), chain of thought (delivering 18% accuracy improvements), validation patterns (preventing 80% of errors), and knowing when to say no. See how quality improvements strengthen the feedback flywheel with a 62% trust score increase.

### Chapter 4: Learning from User Behavior

#### [Chapter 4.1: Finding Patterns in User Data](chapter4-1.md)

How to turn vague feedback into actionable improvements. Learn the difference between topics (what users ask about) and capabilities (what they want done), plus practical clustering techniques. See how a construction company discovered that 8% of queries (scheduling) drove 35% of churn, justifying focused improvement efforts.

#### [Chapter 4.2: Deciding What to Build Next](chapter4-2.md)

Practical prioritization using 2x2 frameworks, failure analysis, and user behavior. See how the construction company chose to fix scheduling (high volume, low satisfaction, clear capability gap) over compliance queries (low volume, already good), driving 35% retention improvement. Real examples of how query analysis changes what you build.

### Chapter 5: Specialized Retrieval That Actually Works

#### [Chapter 5.1: When One Size Doesn't Fit All](chapter5-1.md)

Why generic RAG hits limits and how specialized retrievers solve it. Covers metadata extraction vs. synthetic text strategies and how to measure two-level systems.

#### [Chapter 5.2: Search Beyond Text](chapter5-2.md)

Practical implementations for documents, images, tables, and SQL. Real performance numbers: blueprint search jumping from 16% to 85% recall in four days, vision models bridging the search gap, tables converted to markdown for LLM consumption. Includes decision framework for choosing between summarization, extraction, and RAPTOR approaches.

### Chapter 6: Making It All Work Together

#### [Chapter 6.1: Query Routing Basics](chapter6-1.md)

How to build systems where specialized components work together. See how the construction company improved from 65% to 78% overall success by implementing routing (95% routing accuracy × 82% retrieval quality = 78% end-to-end). Covers team structure, write-time vs read-time compute trade-offs, and the two-level performance formula.

#### [Chapter 6.2: Building the Router](chapter6-2.md)

Practical implementation of routing layers. Learn how few-shot examples drive performance (10 examples: 88% accuracy, 40 examples: 95% accuracy). Includes Pydantic interfaces, structured outputs with Instructor, dynamic examples, and when to use multi-agent vs single-agent designs. See the three-week implementation timeline from basic routing to production-ready system.

#### [Chapter 6.3: Measuring and Improving Routers](chapter6-3.md)

How to know if your router works and make it better. Understand compound effects (67% routing × 80% retrieval = 54% vs 95% × 82% = 78%). Learn dual-mode UIs from Google's approach (specialized interfaces like Maps, Scholar, YouTube), diagnostic frameworks, and setting up improvement loops that feed back to Chapter 1's evaluation framework.

### [Chapter 7: Production Considerations](chapter7.md)

Keeping the improvement flywheel spinning in production. See how the construction company scaled from 500 to 2,500 daily queries while improving from 78% to 84% success and reducing unit costs from $0.09 to $0.04 per query. Covers cost optimization with real dollar amounts, monitoring that connects to Chapter 1's metrics, graceful degradation strategies, and maintaining improvement velocity at scale.

## How These Workshops Work

Each chapter includes practical exercises you can apply to your own RAG system. They build on each other, so start from the beginning unless you know what you're doing.

The progression:

1. **Getting Started** (Intro & Ch 1): Think like a product, set up evaluation
2. **Making It Better** (Ch 2): Turn evaluation into improvements
3. **User Experience** (Ch 3): Get feedback, feel fast, don't break
4. **Learn from Users** (Ch 4): Find patterns, pick what to build
5. **Go Deep** (Ch 5): Build specialized tools that excel
6. **Tie It Together** (Ch 6): Make everything work as one system

## Prerequisites

You should know what RAG is and have at least played with it. If you're totally new, start with the [Introduction](chapter0.md).

## What You'll Have When Done

A RAG system that:

- Gets better from user feedback
- Routes queries to the right specialized tools
- Feels fast and responsive
- Makes improvement decisions based on data
- Handles edge cases gracefully
- Works in production, not just demos
