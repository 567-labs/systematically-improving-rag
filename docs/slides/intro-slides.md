---
title: Systematically Improving RAG — Intro
description: Cohort kickoff — syllabus, logistics, key insights, outcomes
authors:
  - Jason Liu
date: 2025-01-01
theme: seriph
class: text-left
drawings:
  persist: false
highlighter: shiki
lineNumbers: true
download: true
presenter: true
exportFilename: systematically-improving-rag-intro
aspectRatio: 16/9
colorSchema: auto
slideNumber: true
---

# Systematically Improving RAG Applications

November 19, 2025

<br>

By Jason Liu

---

# Introduce yourself!

- Where are you calling in from?
- Are you working on a project right now?
- What are your goals for the course?

---

# Today's Plan

- Introductions
- About me; consulting and training
- Course format & logistics
- Key insights & course outcomes
- Syllabus (Sessions 0–3)
- Syllabus (Sessions 4–6)
- Resources & next steps
- Q&A

---

# About Me (Jason Liu)

- University of Waterloo (2012–2017):
  - Computational Mathematics, Mathematical Physics
  - Computational linear algebra → compression/embedding models → retrieval and deep learning
- Meta (2017):
  - Content policy/moderation, public risk & safety
  - Built dashboards and search tools to surface harmful content
- Stitch Fix (2018–2023):
  - CV + multimodal retrieval
  - VAEs/GANs for GenAI
  - ~$50M incremental revenue
  - Led ~$400K/yr data curation for next‑gen models

---

# Consulting and Training

- Personal note: Hand injury (2021–2022) → shifted focus to higher‑leverage teaching and advising
- Consulting (2023–present): Query understanding, prompts, embedding search, fine‑tuning, MLOps/observability; upgrading legacy workflows to agentic systems
- Clients: HubSpot, Zapier, Limitless, and others across assistants, construction, research
- Recently: helping with medical triage AI; advising startups that do observability

---

# Course Format

**This is a 3-week accelerated version of the course.**

- Inverted classroom: ~5 hours pre‑recorded lectures + tutorial videos + Jupyter exercises
- I recommend watching two sessions per week

---

# Office Hours & Community

- **Office hours**: Bring your problems, introduce yourself
  - Treat it like a tech‑lead review of your work
  - Cameras on is really appreciated! Helps me a lot.
  - Guest lectures: 1-2 times a week, practitioners actively building in the space
- **Slack**: For any questions about code or anything else, please post on the Slack
- **Sharing**: Welcome to share your learnings online via LinkedIn or Twitter (please link back to us somehow)

---

# Logistics & Support

- **Scheduling**: Occasional reschedules; advance notice
- **Credits/support**: Contact Marian — support@jxnl.co
- **Fit concerns**: If you feel like it's not a really good fit for you, just message me and we can figure out how we can make this better for you

---

# Upcoming Talks This Season

- Understanding tool called hallucinations: how LangChain themselves have been thinking about building agents
- Dropbox talk: how they think about knowledge graphs, DSPy, and a bunch of new topics
- Stay tuned to Slack to figure out when these events will happen

---

# What You'll Learn

This course will give you the foundations and practical skills to build, evaluate, and operate retrieval-augmented generation (RAG) systems.

---

# Core Principles

### Keep these in mind

- Good retrieval will often beat clever prompting.
- Similarity is subjective, train for your specific goals.
- Feedback is fuel, design your UX to capture useful signals.
- Specialized indices often outperform one-size-fits-all solutions (though this is shifting as contexts/tools evolve).
- Segmenting queries and users to prioritize work and build a roadmap.
- Production matters, cache, monitor, and degrade gracefully.

---

# The Models are Good but Context Is the Bottleneck

- Models are already very capable for work.
- Even if models hold steady, apps can still improve.
- With the right context, success rates are very high.
- The real challenge is getting that context — the R in retrieval.

---

# Sessions 0–3: Foundations

- **Session 0**: Product mindset; RAG as a recommender; improvement flywheel
- **Session 1**: Synthetic data and retrieval evals; precision/recall; baselines
- **Session 2**: From evals to training data; reranking; embedding fine-tuning
- **Session 3**: UX that collects data; streaming; chain of thought; validation
  - Pacing note: Week 3 is intentionally lighter—use it to catch up and get ahead
  - Focus on UX patterns; not the most critical week content‑wise

---

# Sessions 0–3: Main Takeaway

- Fast retrieval evals (precision/recall on key chunks)
- Rerank/fine‑tune to get a 10-20% improvement
- Deploy and collect real data via UX

---

# Sessions 4–6: Advanced Topics

- **Session 4**: Topic modeling; query segmentation; prioritization frameworks
- **Session 5**: Specialized indices; multimodal search (docs, images, tables, SQL)
- **Session 6**: Query routing; tools-as-APIs; single vs multi-agent; measurement
  - Week 6 is lighter; focus on routing and preview the context‑engineering direction

---

# Sessions 4–6: Main Takeaway

- Figure out what's important to you and your users
- Build specialized indices for those use cases
- Make sure the agent is able to use the specialized indices

---

# Resources

Feel free to share this with coworkers, but don't post these links on social media. You're completely welcome to write your own notes and share them online! (Please link back to us somehow)

![QR Codes for Resources](./assets/images/codes.jpeg)

- Study notes (work in progress): https://567-labs.github.io/systematically-improving-rag/
- Talks/"greatest hits": https://567-labs.github.io/systematically-improving-rag/talks/
- Slack: https://join.slack.com/t/improvingrag/shared_invite/zt-3dkinqb3q-vknvaBLoTx5tBj4PpGOVjw
- Contribute via PRs/issues; add examples; suggest edits

---

# Recommended Talks

- Skylar's RAG anti‑patterns Talk
- Anton's Text Chunking Strategies Talk
- Exa's Why Google Search Sucks for AI Talk
- Colin's Agentic RAG Talk

---

# Q&A and Next Steps

Short Q&A (About the Class Format) and then we'll let you watch the first videos

1. Find 'Optional: Watch Lecture' on your Calendar as a shortcut
2. Feel free to watch at 2x speed!

See you at the office hours!
