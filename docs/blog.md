---
authors:
  - jxnl
categories:
  - RAG
comments: true
date: 2024-05-22
description:
  Discover systematic strategies to enhance your Retrieval-Augmented Generation
  (RAG) systems for better performance and user experience.
draft: false
tags:
  - RAG
  - AI
  - Machine Learning
  - Data Retrieval
  - Performance Optimization
---

# Systematically Improving Your RAG

## The Problem That Started It All

A legal tech company launched their case law search with 63% accuracy. Users were frustrated—missing relevant precedents, getting unrelated cases. The engineering team tried three different embedding models and tweaked prompts dozens of times. Nothing helped.

The fundamental issue? Everyone was treating RAG as a one-time implementation project rather than an evolving product. They'd optimize for the wrong metrics, guess at solutions, and make random changes hoping something would stick.

Three months later, through systematic measurement and improvement, they reached 87% accuracy. User trust scores increased 62%. They generated 50,000+ citation examples for continuous training. The difference? They adopted a product mindset with continuous feedback loops.

This transformation didn't come from finding the perfect model or magical prompt. It came from systematic improvement guided by data.

## The Two Biases That Kill RAG Projects

Behind these surface-level mistakes lie two fundamental biases that kill more RAG projects than anything else:

### Absence Bias (Absence Blindness)

You can't fix what you can't see. Teams obsess over generation quality while ignoring whether retrieval works at all.

Real example: A construction company spent three weeks optimizing prompts for blueprint search. When they finally checked retrieval metrics, they found only 27% recall—the system wasn't even finding the right blueprints. No amount of prompt engineering could fix that.

After switching focus to retrieval quality through vision model summaries, they reached 85% recall in four days. Later, they discovered 20% of queries involved counting objects, justifying investment in bounding box models that pushed counting accuracy to 92%.

Questions teams forget to ask:

- Is retrieval actually finding the right documents?
- Are our chunks the right size?
- Is our data extraction pipeline working?
- Do we have separate metrics for retrieval versus generation?

### Intervention Bias

This is our tendency to do _something_ just to feel like we're making progress. In RAG, it shows up as constantly switching models, tweaking prompts, or adding features without measuring impact.

- Should we use GPT-4 or Claude? Maybe a new embedding model? What about chunking?
- Will this new prompt technique help? I read on Arxiv that this new prompt technique is the best...

The solution? Resist the hype. Start from real data, measure carefully, do simple experiments, and only keep what actually helps.

!!! quote "You Can't Manage What You Can't Measure"

    This classic management principle applies perfectly to RAG systems. Without proper measurement, you're flying blind—making changes based on hunches rather than data. The teams that succeed are the ones who measure everything: retrieval quality, generation accuracy, user satisfaction, and system performance.

## My Systematic RAG Playbook

This is my systematic approach to making RAG applications better. It's a step-by-step guide that will help you improve your RAG systems over time.

By the end of this post, you'll understand my systematic approach to making RAG applications better. We'll look at important areas like:

- Making fake questions and answers to quickly test how well your system works
- Using both full-text search and vector search together for the best results
- Setting up the right ways to get feedback from users about what you want to study
- Using grouping to find sets of questions that have problems, sorted by topics and abilities
- Building specific systems to improve abilities
- Constantly checking and testing as you get more real-world data

Through this step-by-step runbook, you'll gain practical knowledge on how to incrementally enhance the performance and utility of your RAG applications, unlocking their full potential to deliver exceptional user experiences and drive business value. Let's dive in and explore how to systematically improve your RAG systems together!

<!-- more -->

## 1) Start with Synthetic Data

Here's a brutal truth: most teams spend months fine-tuning prompts without ever measuring if their retrieval actually works. When they finally check, they discover their systems retrieve the correct documents only 30-40% of the time.

No amount of prompt engineering can fix retrieving the wrong information.

**Synthetic data** is your escape hatch. It lets you evaluate your system before you have real users, so you can fix the fundamentals first.

### How to Create Synthetic Data

1. Create synthetic questions for each text chunk in your database
2. Use these questions to test your retrieval system
3. Calculate precision and recall scores to establish a baseline
4. Identify areas for improvement based on the baseline scores

**What is Recall?** Recall measures how many of the relevant documents your system actually found. If there are 10 relevant documents for a query and your system only found 7 of them, your recall is 70%. In RAG systems, missing relevant documents is often worse than finding some irrelevant ones, so recall is usually the more critical metric.

What we should be finding with synthetic data is that synthetic data should just be around 80-90% recall. And synthetic data might just look like something very simple to begin with.

A simple procedure works but only if you look at your own data. Imagine for every text chunk, I want it to synthetically generate a set of questions that this text chunk answers. For those questions, can we retrieve those text chunks? And you might think the answer is always going to be yes. But I found in practice that when I was doing tests against essays, full text search and embeddings basically performed the same, except full text search was about 10 times faster.

Whereas when I did the same experiment on pulling issues from a repository, it was the case that full text search got around 55% recall, and then embedding search got around 65% recall. Way worse! And just knowing how challenging these questions are on the baseline is super important to figure out what kind of experimentation you need to perform better.
This will give you a baseline to work with and help you identify areas for improvement.

!!! info "Agent Persistence Changes Everything"

    State-of-the-art coding agents have reached the top of SWE-Bench leaderboards using simple grep and find tools instead of sophisticated embeddings. Why? Agent persistence compensates for less sophisticated tools - they'll keep trying different approaches until they find what they need. This means traditional component-level metrics (like embedding quality) may not translate to end-to-end performance in agent-based systems.

This doesn't mean that we need an agentic rag for everything. Inventing search is cheap and can be seen as a cost-saving factor, both in terms of maintenance, implementation, and complexity. What Semantic Search could do with one API call, Agentic Rag could do with 15 if you give it simpler tools. Again, we're making trade-offs along a continuum.

But the benefit is - if you have a number to set a goal against, you can always trade off your recall, your latency, and your performance.

## 2) Utilize Metadata

Your users ask questions that pure text search can't answer. "What's the latest version?" "Show me documents from last quarter." "Who owns this project?"

**Metadata** bridges this gap. It's the structured information that makes your documents searchable beyond just their text content.

Think of it as adding labels to your documents: dates, authors, categories, statuses. When users ask specific questions, you can filter and find exactly what they need.

For example, if someone asks, "What is the latest x, y, and z?" Text search will never get that answer. Semantic search will never get that answer.

You need to perform query understanding to extract date ranges. There will be some prompt engineering that needs to happen. That's the metadata, and being aware that there will be questions that people aren't answering because those filters can never be caught by full text search and semantic search.

And what this looks like in practice is if you ask the question, what are recent developments in the field, the search query is now expanded out to more terms. There's a date range where the language model has reasoned about what recent looks like for the research, and it's also decided that you should only be searching specific sources. If you don't do this, then you may not get trusted sources. You may be unable to figure out what recent means.

You'll need to do some query understanding to extract date ranges and include metadata in your search.

This has become more and more feasible as function calling and other loans become much more powerful. You can just have arguments that include start dates and end dates and do quite well. But what this still means is you need to index things correctly with the relevant metadata.

## 3) Use Both Full-Text Search and Vector Search

The RAG community loves to debate: "Should I use full-text search or vector search?"

Here's the secret: **use both**. They solve different problems, and the best RAG systems combine them intelligently.

Full-text search finds exact matches and handles technical terms. Vector search understands meaning and context. Together, they cover the full spectrum of what users actually ask.

### Understanding the Differences

**Full-Text Search (Lexical Search):**

- Uses traditional information retrieval techniques like BM25 scoring
- Excels at exact matches for product codes, names, and specific phrases
- Handles technical terms and abbreviations that embedding models might miss
- Provides efficient filtering and metadata aggregation
- Works well with Boolean logic and structured queries
- Much faster for simple keyword searches

**Vector Search (Semantic Search):**

- Converts text into embedding vectors that capture meaning
- Finds semantically similar content even when exact words don't match
- Handles synonyms and different ways of expressing the same concept
- Better at understanding context and intent
- Can process complex, contextual queries
- More effective for conceptual and exploratory searches

Okay, so which one do I use?

It depends on your data. This is why you created synthetic data in the first place. Don't take my word. Set up evals and check whether or not semantic search is worth the 10x greater latency than full text search based on the recall. If you get a 1% improvement with 10x latency, maybe it's not worth it. If you get a 15% improvement in recall, maybe it is. It's up to you. It's up to your problem.

## 4) Implement Clear User Feedback Mechanisms

Most RAG systems get feedback rates below 1%. Users don't tell you what's wrong because you're not asking the right questions.

**Good feedback collection** means asking specific, actionable questions. "Did we answer your question correctly?" beats "How did we do?" every time.

The goal is to build a continuous improvement loop where user feedback drives your next improvements.

I find that it's important to build out these feedback mechanisms as soon as possible. And making sure that the copy of these feedback mechanisms explicitly describe what you're worried about.

Sometimes, we'll get a thumbs down even if the answer is correct, but they didn't like the tone. Or the answer was correct, but the latency was too high. Or it took too many hops.

This means we couldn't actually produce an evaluation dataset just by figuring out what was a thumbs up and a thumbs down. It was a lot of confounding variables. We had to change the copy to just "Did we answer the question correctly? Yes or no." We need to recognize that improvements in tone and improvements in latency will come eventually. But we needed the user feedback to build us that evaluation dataset.

Make sure the copy for these feedback mechanisms explicitly describes what you're worried about. This will help you isolate the specific issues users are facing.

## 5) Cluster and Model Topics

You've collected user feedback and now have thousands of queries. Great! But staring at individual ratings won't tell you what to fix next after a few 1000 examples.

You need to see the bigger picture.

**Topic modeling and clustering** transform your raw feedback into actionable insights. Instead of reading through queries one by one, you group similar ones and look for patterns. This reveals the real problems worth fixing.

The key insight? Not all improvements matter equally. Some query types affect 80% of your users, while others might be rare but critical for your biggest customers. You need to know the difference.

Consider this example from a technical documentation search system. By clustering user queries, two main patterns emerged:

1. Topic Clusters: A significant portion of user queries were related to a specific product feature that had recently been updated. However, our system was not retrieving the most up-to-date documentation for this feature, leading to confusion and frustration among users.

2. Capability Gaps: Another cluster of queries revealed that users were frequently asking for troubleshooting steps and error code explanations. While our system could retrieve relevant documentation, it struggled to provide direct, actionable answers to these types of questions.

Based on these insights, we prioritized updating the product feature documentation and implementing a feature to extract step-by-step instructions and error code explanations. These targeted improvements led to higher user satisfaction and reduced support requests.

Look for patterns like:

- Topic clusters: Are users asking about specific topics more than others? This could indicate a need for more content in those areas or better retrieval of existing content.

- Capabilities: Are there types of questions your system categorically cannot answer? This could indicate a need for new features or capabilities, such as direct answer extraction, multi-document summarization, or domain-specific reasoning.

By continuously analyzing topic clusters and capability gaps, you can identify high-impact areas for improvement and allocate your resources more effectively. This data-driven approach to prioritization ensures that you're always working on the most critical issues affecting your users.

Once you have this in place, once you have these topics and these clusters, you can talk to domain experts for a couple of weeks to figure out what these categories are explicitly. Then, you can build out systems to tag that as data comes in.

In the same way that when you open up ChatGPT and make a conversation, it creates an automatic title in the corner. You can now do that for every question. As part of that capability, you can add the classification, such as what are the topics and what are the capabilities. Capabilities could include ownership and responsibility, fetching tables, fetching images, fetching documents only, no synthesis, compare and contrast, deadlines, and so on.

You can then put this information into a tool like Amplitude or Sentry (or Raindrop for ai natives). This will give you a running stream of the types of queries people are asking, which can help you understand how to prioritize these capabilities and topics.

## 6) Continuously Monitor and Experiment

Your RAG system isn't a one-time project. It's a living system that needs constant attention.

**Continuous monitoring** means tracking performance over time, not just during development. You need to know when things break, when performance degrades, and when new patterns emerge.

The key is running experiments systematically. Test changes on real data, measure the impact, and only keep what actually helps.

This could include tweaking search parameters, adding metadata, or trying different embedding models. Measure the impact on precision and recall to see if the changes are worthwhile.

!!! quote "Start with Vibe Checks, Not Just Metrics"
"Start with 5-10 examples and do end-to-end vibe checks before moving to quantitative evaluation. With natural language systems, you can learn so much just from looking at a few examples." This is especially important for agent-based systems where improving individual components (like embeddings) might not improve end-to-end performance.

Once you now have these questions in place, you have your synthetic data set and a bunch of user data with ratings. This is where the real work begins when it comes to systematically improving your RAG.

The system will be running many clusters of topic modeling around the questions, modeling that against the thumbs up and thumbs down ratings to figure out what clusters are underperforming. It will then determine the count and probability of user dissatisfaction for each cluster.

The system will be doing this on a regular cadence, figuring out for what volume of questions and user satisfaction levels it should focus on improving these specific use cases.

What might happen is you onboard a new organization, and all of a sudden, those distributions shift because their use cases are different. That's when you can go in and say, "We onboarded these new clients, and they very much care about deadlines. We knew we decided not to service deadlines, but now we know this is a priority, as it went from 2% of questions asking about deadlines to 80%." You can then determine what kind of education or improvements can be done around that.

## 7) Balance Latency and Performance

Every improvement comes with a cost. Better recall might mean slower responses. More sophisticated search might mean higher compute costs.

**Trade-off decisions** should be based on your specific use case, not generic best practices. A medical diagnostic tool might prioritize accuracy over speed, while a customer support chatbot might need instant responses.

The key is measuring the impact of each change and making informed decisions about what matters most for your users.

!!! tip "Agentic Systems Change the Equation"
In agentic systems, the trade-offs shift. Agents are persistent - they'll eventually find what they need even with suboptimal tools. This means you might accept higher latency for better end-to-end outcomes, since agents can iterate and course-correct. The best approach often combines both: expose your high-quality embeddings as tools to agents, getting persistence AND efficiency.

Here, this is where having the synthetic questions that test against will effectively answer that question. Because what we'll do is we'll run the query with and without this parent document retriever, and we will have a recall with and without that feature and the latency improvement of that feature.

And so now we'll be able to say, okay. Well, recall doubles. The latency increases by 20%, then a conversation can happen. Or, is that worth the investment? But if latency goes up double and the recall goes up 1%, again, it depends on, okay.

Well, if this is a medical diagnostic, maybe I do care that the 1% is included because the stakes are so high. But if it's for a doc page, maybe the increased latency will reduce in churn.

If you can improve recall by 1%, and the system is too complex, it's not worth deploying it in the future as well.

For example, if you're building a medical diagnostic tool, a slight increase in latency might be worth it for better recall. But if you're building a general-purpose search tool, faster results might be more important.

## Wrapping Up

This systematic approach has been refined through countless implementations. While many implementation details have been simplified here, the key is to start measuring and iterating rather than guessing.

## Want to learn more?

For a deeper dive into these concepts, check out the free 6-week email course on RAG that covers all these strategies in detail:

[Enroll in the Free 6-Day Email Course](https://improvingrag.com/){ .md-button .md-button--primary }

---
