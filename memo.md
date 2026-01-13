# Data Analysis Arena: Project Memo

## Vision

A crowdsourced benchmark platform that evaluates AI systems' ability to perform complete data analysis tasks - not just code generation, but the full workflow of exploring data, generating insights, creating visualizations, and communicating findings effectively.

Similar to Design Arena's approach for visual design, but focused on data analysis capabilities across the entire pipeline.

## Core Concept

**Input:** User uploads a dataset (CSV, SQLite, JSON, etc.)

**Process:** Multiple AI systems independently analyze the data and produce deliverables

**Output:** Each system generates analysis in various formats (Jupyter notebooks, Streamlit dashboards, Quarto reports, etc.)

**Evaluation:** Human voters compare outputs side-by-side and vote on which analysis is more useful

## What Makes This Different

Current code benchmarks (HumanEval, MBPP) only test:
- Syntax correctness
- Algorithm implementation
- Isolated function behavior

**Data Analysis Arena tests:**
1. **Python Ability** - Data wrangling, statistical analysis, code quality
2. **Communication** - Clear insights, actionable recommendations, storytelling
3. **Visualization** - Chart selection, interactivity, design aesthetics

This evaluates the complete skill set needed to replace/augment a data analyst.

## AI Systems to Compare

### Large Language Model APIs (With Computer Use APIs)

- GPT 5 (Responses API)
- Claude
- Gemini
- OpenRouter
- XAI

### AI Coding Agents (In a Docker Container and CLI Tools)
- Claude Code SDK CLI
- Gemini SDK CLI
- OpenCode SDK CLI
- Codex SDK CLI
- Devin SDK CLI

Each system operates independently with its own approach to solving the analysis task.

## Input Formats

### MVP (Phase 1)
- **CSV** - Universal tabular data format
- **SQLite** - Multi-table relational data
- **JSON/JSONL** - Semi-structured nested data
- **Excel** - Business-standard spreadsheets

## Output Formats

### MVP (Phase 1)
- **Jupyter Notebooks** - Interactive code + outputs, educational format
- **Streamlit** - Interactive dashboards, quick prototypes
- **Quarto/HTML** - Publication-quality reports
- **CSV/Excel** - Clean data artifacts with summaries

## The Three-Pillar Scoring System

Each analysis is evaluated across three dimensions:

### 1. Python Ability (Technical Execution, does the code run?)

- Data cleaning and transformation
- Statistical methods and modeling
- Code efficiency and structure
- Error handling
- Appropriate library usage

### 2. Communication (Storytelling, does the analysis make sense?)

- Clear narrative flow
- Actionable insights ("so what?")
- Executive summary
- Context and recommendations
- Audience-appropriate depth

### 3. Plotting/Interactivity (Visualization, does the analysis look good?)

- Appropriate chart types
- Clear labels and legends
- Interactive elements (filters, tooltips)
- Design aesthetics
- Information density

**Key insight:** All three must be strong for a great analysis. Technical excellence without clear communication is useless. Beautiful visualizations without correct analysis are misleading.

## Example Arena Match

```
Dataset: E-commerce transactions (50K rows)
Task: "Analyze customer behavior and identify revenue opportunities"
Output: Jupyter Notebook

System A: Claude Code CLI → Jupyter Notebook
System B: GPT-4o-mini (Responses API) → Jupyter Notebook

Voters choose: System A
```

## Dataset Characteristics to Test

### Domains and benchmarks from a bunch of different datasets + custom uploaded datasets

- **E-commerce:** Sales, customers, products
- **SaaS:** User behavior, churn, retention
- **Finance:** Transactions, time series, risk
- **Text:** Reviews, sentiment, topic modeling
- **Time Series:** Stocks, weather, metrics
- **Healthcare:** (later phase - privacy concerns)

## Technical Architecture

### Backend Pipeline
```
1. User selects dataset/ uploads dataset + task
2. System spawns parallel jobs for n AI systems
3. Each system:
   - Receives dataset + prompt
   - Performs analysis autonomously (container, computer use, etc.)
   - Generates output in assigned format
   - Stores results, throws out the uploaded dataset
4. Frontend displays results side-by-side
5. Users vote on preference
6. Results update leaderboard (ELO/win rate)
```

### Execution Environment Requirements

For systems to truly perform analysis (not just generate code), we need:

**Jupyter Kernel Management Tool perhaps there is a trusted MCP for this**
- `StartKernel(notebook_path)` - Initialize persistent kernel
- `ListCells(kernel_id)` - View all cells
- `ViewCell(kernel_id, cell_id)` - See specific cell + outputs
- `EditCell(kernel_id, cell_id, new_source)` - Modify cell
- `InsertCell(kernel_id, after_cell_id, source)` - Add new cell
- `RunCell(kernel_id, cell_id)` - Execute single cell
- `RunCellsBelow(kernel_id, cell_id)` - Execute from point onward
- `GetVariables(kernel_id)` - Inspect kernel state
- `StopKernel(kernel_id)` - Clean shutdown

### Libraries to Install

**Core Data Manipulation:**
- pandas - Dataframe operations
- numpy - Numerical computing
- polars - Fast dataframe alternative
- scipy - Scientific computing

**Visualization:**
- matplotlib - Basic plotting
- seaborn - Statistical visualizations
- plotly - Interactive plots
- altair - Declarative visualizations
- bokeh - Interactive web visualizations

**Machine Learning:**
- scikit-learn - Classical ML algorithms
- xgboost - Gradient boosting
- lightgbm - Fast gradient boosting
- statsmodels - Statistical modeling
- prophet - Time series forecasting

**NLP & Embeddings:**
- sentence-transformers - Semantic similarity, embeddings
- transformers - BERT, GPT, and other transformer models
- nltk - Natural language toolkit
- spacy - Industrial-strength NLP
- gensim - Topic modeling and word embeddings

**App Frameworks:**
- streamlit - Quick data apps
- gradio - ML interfaces
- dash - Plotly dashboards
- panel - Custom apps

**Dataset Libraries:**
- scikit-learn - Built-in datasets (Iris, Digits, Wine, Breast Cancer, California Housing)
- seaborn - Statistical datasets (Titanic, Tips, Flights, Diamonds, Iris)
- datasets (HuggingFace) - Access to thousands of datasets
- kaggle - Kaggle dataset API
- ucimlrepo - UCI Machine Learning Repository

**Data Quality & Profiling:**
- ydata-profiling - Automated EDA
- great-expectations - Data validation

**Notebook & Utilities:**
- jupyter - Interactive notebooks
- ipywidgets - Interactive widgets
- nbformat - Notebook format handling

### Common Datasets Available

**Built-in from scikit-learn:**
- Iris - Classification (150 rows, 4 features)
- Wine - Classification (178 rows, 13 features)
- Breast Cancer - Binary classification (569 rows, 30 features)
- Diabetes - Regression (442 rows, 10 features)
- California Housing - Regression (20K rows, 8 features)
- Digits - Image classification (1797 images)

**Built-in from seaborn:**
- Titanic - Survival classification (891 rows)
- Tips - Regression/analysis (244 rows)
- Flights - Time series (144 rows)
- Diamonds - Regression (53K rows)
- MPG - Auto dataset (398 rows)

**Popular Kaggle Datasets:**

Classification Tasks:
- Titanic - Survival prediction (891 rows, classic binary classification)
- Adult Income - Predict income >50K (48K rows, demographic features)
- Bank Marketing - Predict term deposit subscription (45K rows)
- Credit Card Fraud Detection - Imbalanced classification (285K rows)
- Wine Quality - Multi-class quality rating (6.5K rows)
- HR Analytics - Employee attrition prediction (15K rows)
- Customer Churn - Telecom churn prediction (7K rows)

Regression Tasks:
- House Prices (Ames Housing) - Price prediction (1.5K rows, 80 features)
- California Housing - Median house values (20K rows)
- Bike Sharing Demand - Count prediction (17K rows, time series)
- Used Car Prices - Vehicle valuation (3M rows)
- Insurance Costs - Medical cost prediction (1.3K rows)

Text Analysis:
- IMDB Movie Reviews - Sentiment analysis (50K reviews)
- Spam Classification - Email spam detection (5.5K messages)
- Amazon Product Reviews - Multi-category sentiment (4M reviews)
- News Category Classification - Topic modeling (200K articles)
- Fake News Detection - Binary classification (20K articles)

Time Series:
- COVID-19 Dataset - Daily cases worldwide (evolving)
- Stock Market Data - Historical prices (various tickers)
- Store Sales Forecasting - Retail time series (125K rows)
- Web Traffic Forecasting - Wikipedia pageviews (145K series)
- Energy Consumption - Household power usage (2M measurements)

Multi-table/Relational:
- Instacart Market Basket Analysis - 3M orders, 6 tables
- Walmart Store Sales - 421K rows, multiple stores
- Retail Data Analytics - Sales across stores and products

Visual/Rich Content:
- Netflix Movies & TV Shows - Content catalog (8.8K titles)
- YouTube Trending Videos - Video metadata (200K videos)
- Top 1000 IMDb Movies - Movie analytics (1K movies with rich metadata)
- Spotify Song Attributes - Music analysis (170K tracks)

**HuggingFace Datasets (Tabular):**

- `scikit-learn/*` - All sklearn datasets (iris, wine, diabetes, etc.)
- `inria-soda/tabular-benchmark` - Curated ML benchmarks
- `mstz/adult` - Adult income dataset
- `mstz/wine_quality` - Wine quality (red and white)
- `scikit-learn/california-housing` - California housing
- `Harrison/california-housing` - Alternative California housing
- `csv-datasets/*` - Various CSV datasets

**HuggingFace Datasets (Text for Analysis):**

- `imdb` - Movie reviews (50K, sentiment)
- `yelp_review_full` - Yelp reviews (650K, 5-star rating)
- `amazon_polarity` - Amazon reviews (3.6M, binary sentiment)
- `emotion` - Twitter emotions (20K, 6 emotions)
- `financial_phrasebank` - Financial sentiment (5K sentences)
- `rotten_tomatoes` - Movie reviews (11K)

**Data Journalism Sources:**
- FiveThirtyEight Data - Politics, sports, entertainment (well-documented CSVs)
- Pew Research Center - Survey data (demographic analysis)
- Data.gov - US government open data (thousands of datasets)
- World Bank Open Data - Global development indicators
- Our World in Data - Research datasets on global issues
- ProPublica Data Store - Investigative journalism data

### Library Usage Tracking and Analysis

- Provide systems with recommended libraries in the prompt
- Example: "You have pandas, numpy, matplotlib, seaborn, plotly, scikit-learn available"

**Metrics to Track:**
- Library usage frequency (which libraries are used most)
- Library combinations (pandas + plotly vs pandas + matplotlib)
- Library choice vs vote outcomes (does plotly correlate with better viz scores?)
- Library choice vs execution time (is polars actually faster in practice?)
- Library choice vs code quality (fewer lines, cleaner code)

**Interesting Research Questions:**
- Do interactive visualizations (plotly, altair) beat static (matplotlib)?
- Does polars usage correlate with higher Python scores for large datasets?
- Which ML library choice wins for different problem types?
- Do simpler library stacks (fewer imports) get better scores?
- Does seaborn lead to better design aesthetics than raw matplotlib?

**Leaderboard Breakdowns:**
```
Best Library Choices by Category:

Visualization:
- Interactive dashboards: plotly (ELO +45 vs matplotlib)
- Statistical plots: seaborn (ELO +23 vs matplotlib)
- Publication quality: matplotlib + seaborn (ELO +15)

Data Manipulation:
- Small datasets (<100K): pandas (baseline)
- Large datasets (>1M): polars (ELO +12, 3x faster)
- Multi-table: pandas (most common, well-understood)

Machine Learning:
- Classification: scikit-learn + xgboost combo (ELO +18)
- Time series: prophet or statsmodels (ELO +25)
- Quick baselines: scikit-learn only (most reliable)
```

**Implementation:**
- Parse imports from generated code
- Track library versions used
- Record execution time per library combination
- Store in metadata for each battle
- Generate library usage reports on leaderboard

### Safety & Cost Controls

- Sandboxed execution (Docker containers)
- Time limits (10 min max per analysis)
- Token budgets per analysis
- Rate limiting per user
- No network access from execution environment
- Resource limits (CPU, memory, disk)

## Voting & Evaluation

```Input View - Modern UI Design

┌─────────────────────────────────────────────────────────────────────────────┐
│                          🔬 Data Analysis Arena                             │
│                     Compare AI Systems on Real Analysis                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  📊 Choose Your Dataset                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐  ┌─────────────────────────┐                  │
│  │     📁 Upload Dataset   │  │   📚 Select Curated     │                  │
│  │                         │  │                         │                  │
│  │   Drag & drop files     │  │    Browse examples      │                  │
│  │   or click to browse    │  │   • E-commerce Sales    │                  │
│  │                         │  │   • Customer Churn     │                  │
│  │   Supported formats:    │  │   • Product Reviews    │                  │
│  │   • CSV, Excel          │  │   • Stock Prices       │                  │
│  │   • SQLite, JSON        │  │   • Web Analytics      │                  │
│  │   • Up to 100MB         │  │                         │                  │
│  └─────────────────────────┘  └─────────────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  🎯 Analysis Output Format                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ 📓 Jupyter   │ │ 📊 Streamlit │ │ 📄 Quarto    │ │ 📋 Data      │      │
│  │   Notebook   │ │  Dashboard   │ │   Report     │ │  Export      │      │
│  │              │ │              │ │              │ │              │      │
│  │ Interactive  │ │ Interactive  │ │ Publication  │ │ Clean CSV/   │      │
│  │ code & viz   │ │ web app      │ │ quality      │ │ Excel files  │      │
│  │              │ │              │ │              │ │              │      │
│  │   ○ Select   │ │   ○ Select   │ │   ○ Select   │ │   ○ Select   │      │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  ⚙️ Analysis Settings                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Analysis Goal: ┌─────────────────────────────────────────────────────────┐ │
│                 │ Describe what insights you're looking for...            │ │
│                 │ e.g., "Find revenue opportunities and customer trends"  │ │
│                 └─────────────────────────────────────────────────────────┘ │
│                                                                             │
│  AI Systems:    ☑ Claude 3.5 Sonnet    ☑ GPT-4                            │
│                 ☑ Gemini Pro           ☐ Claude Code CLI                   │
│                                                                             │
│  Time Limit:    ○ 5 min (fast)  ● 10 min (standard)  ○ 15 min (thorough) │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────────────────┐
                        │    🚀 Start Analysis        │
                        │      Battle Arena           │
                        └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  💡 What happens next?                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Multiple AI systems analyze your data simultaneously                    │
│  2. Each creates a complete analysis in your chosen format                  │
│  3. You'll see results side-by-side for comparison                         │
│  4. Vote on which analysis is more useful                                  │
│  5. Help improve AI systems through your feedback                          │
└─────────────────────────────────────────────────────────────────────────────┘

```



### Voting Interface - Battle View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📊 Data Analysis Arena                    Battle #1,247      [Full Leaderboard →]  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Dataset: E-commerce Sales (50K rows)      Format: Jupyter Notebook        │
│  Task: "Analyze customer behavior and identify revenue opportunities"      │
│  ⏱️ Just now  •  📋 CSV  •  🎯 Business Intelligence                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────┬───────────────────────────────────────┐
│                                   │                                       │
│         🤖 System A               │           🤖 System B                 │
│       Claude Code CLI             │            GPT-4o                     │
│                                   │                                       │
│      ⚡ 1368 ELO  •  68.8% WR     │      ⚡ 1320 ELO  •  67.3% WR         │
│                                   │                                       │
├───────────────────────────────────┼───────────────────────────────────────┤
│                                   │                                       │
│                                   │                                       │
│   ┌─────────────────────────┐    │    ┌─────────────────────────┐       │
│   │                         │    │    │                         │       │
│   │   📊 Revenue Analysis   │    │    │   📈 Customer Segments  │       │
│   │                         │    │    │                         │       │
│   │   [Interactive chart]   │    │    │   [Interactive chart]   │       │
│   │   • 3 visualizations    │    │    │   • 4 visualizations    │       │
│   │   • 5 insights          │    │    │   • 3 insights          │       │
│   │   • Clear narrative     │    │    │   • Statistical depth   │       │
│   │                         │    │    │                         │       │
│   └─────────────────────────┘    │    └─────────────────────────┘       │
│                                   │                                       │
│   [View Full Notebook →]         │    [View Full Notebook →]             │
│                                   │                                       │
└───────────────────────────────────┴───────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Which analysis is better?                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│         ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│         │      A       │    │     Tie      │    │      B       │          │
│         │   ← Better   │    │   Equal ≈    │    │  Better →    │          │
│         └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                             │
│         [Keyboard: A]       [Keyboard: T]       [Keyboard: B]              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  💭 What made it better? (optional - helps improve AI systems)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Python & Code Quality          Communication                              │
│  ☐ Better data cleaning         ☐ Clearer insights                        │
│  ☐ More appropriate methods     ☐ Better storytelling                     │
│  ☐ Efficient code               ☐ Actionable recommendations               │
│  ☐ Error handling               ☐ Executive summary                        │
│                                                                             │
│  Visualization & Design         Overall                                    │
│  ☐ Better chart selection       ☐ More thorough analysis                  │
│  ☐ Clearer labels               ☐ Easier to understand                    │
│  ☐ Good interactivity           ☐ Better presentation                     │
│  ☐ Professional aesthetics      ☐ More useful for decisions               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                   ┌──────────────┐        ┌──────────────┐
                   │    Submit    │        │  Skip Vote   │
                   └──────────────┘        └──────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  🎯 Your Impact: 47 votes today  •  892 total  •  Top 5% contributor       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Leaderboard Interface

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🏆 Data Analysis Arena Leaderboard                                         │
│  Join 127,482 voters to discover which AI is best at data analysis          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  ┌──────────────┐  ┌──────────────┐                                         │
│  │ All Systems  │  │   Filter ⌄   │      Range: [All] [14d] [30d] [90d]   │
│  └──────────────┘  └──────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Rank  Model                    ELO Rating ↓    Win Rate    MoE    Battles │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  #1   🔸 Claude Sonnet 4.5         1368         68.8%      ±6.2%    215   │
│       (Thinking Mode)           148W / 67L                                  │
│                                                                             │
│       [Python: 92%] [Communication: 89%] [Visualization: 85%]              │
│       Best at: E-commerce, Time Series, Text Analysis                      │
│       Avg Time: 1m 25s  •  Anthropic                                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  #2   🔸 Claude Opus 4             1349         71.7%      ±1.6%   3,083   │
│       (Standard)                2210W / 873L                                │
│                                                                             │
│       [Python: 88%] [Communication: 91%] [Visualization: 83%]              │
│       Best at: Customer Analytics, Forecasting                             │
│       Avg Time: 1m 31s  •  Anthropic                                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  #3   ⭕ GPT-5 (Minimal)           1320         67.3%      ±2.8%   1,068   │
│       (Fast Mode)                719W / 349L                                │
│                                                                             │
│       [Python: 85%] [Communication: 82%] [Visualization: 88%]              │
│       Best at: Dashboards, Interactive Viz                                 │
│       Avg Time: 2m 0s  •  OpenAI                                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  #4   🔷 Gemini Pro 2.0            1305         66.1%      ±3.1%     847   │
│       (Experimental)             560W / 287L                                │
│                                                                             │
│       [Python: 84%] [Communication: 85%] [Visualization: 79%]              │
│       Best at: Multi-table Analysis, SQL                                   │
│       Avg Time: 1m 48s  •  Google                                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  #5   🔸 Claude Code CLI           1298         70.2%      ±4.2%     412   │
│       (Agent Mode)               289W / 123L                                │
│                                                                             │
│       [Python: 91%] [Communication: 78%] [Visualization: 82%]              │
│       Best at: Complex Workflows, Iterative Analysis                       │
│       Avg Time: 3m 15s  •  Anthropic                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                          [Show 15 More Systems ↓]

┌─────────────────────────────────────────────────────────────────────────────┐
│  📊 Performance by Category                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [All Categories] [E-commerce] [Time Series] [Text] [Customer Analytics]  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ELO Rating                                                         │   │
│  │                                                                     │   │
│  │  1400 │████                                                         │   │
│  │       │████                                                         │   │
│  │  1350 │████ ███                                                     │   │
│  │       │████ ███ ███                                                 │   │
│  │  1300 │████ ███ ███ ██ ██ ██ ██ ██ ██ ██                           │   │
│  │       └─────┴───┴───┴──┴──┴──┴──┴──┴──┴──                           │   │
│  │        CS4  CO4 G5  GP2 CL  XAI DS  O1  C37 GLM                     │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  🎯 Specialty Rankings                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Best at Python              Best at Communication    Best at Visualization│
│  1. Claude Opus 4 (91%)      1. Claude Opus 4 (91%)   1. GPT-5 (88%)      │
│  2. Claude Code CLI (91%)    2. Claude S4.5 (89%)     2. Claude S4.5 (85%)│
│  3. Claude S4.5 (92%)        3. Gemini Pro (85%)      3. Claude Opus (83%)│
│                                                                             │
│  Best Output Format          Fastest Analysis         Most Battles         │
│  Jupyter: Claude Opus 4      Claude S4.5 (1m 25s)     Claude Opus 4       │
│  Streamlit: GPT-5            GPT-5 (2m 0s)            (3,083 battles)      │
│  Quarto: Claude S4.5         Gemini Pro (1m 48s)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  ℹ️ About the Rankings                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ELO Rating: Skill-based rating starting at 1200. Calculated using the     │
│  Bradley-Terry model based on head-to-head voting results.                 │
│                                                                             │
│  Margin of Error: Win rates show ±margin of error based on battle count    │
│  for an approximate 95% Wilson score confidence interval.                  │
│                                                                             │
│  Three-Pillar Scoring: Each system is evaluated on Python ability,         │
│  communication quality, and visualization effectiveness.                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Recent Battles

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📈 Recent Prompts and Battles                    [View Full Tournament →] │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│                  │                  │                  │                  │
│  ⏱️ Just now      │  ⏱️ 5m ago       │  ⏱️ 12m ago      │  ⏱️ 18m ago      │
│  📋 CSV          │  📊 SQLite       │  📝 JSON         │  📋 Excel        │
│                  │                  │                  │                  │
│  E-commerce      │  Customer Churn  │  Product Reviews │  Sales Pipeline  │
│  Revenue Trends  │  Prediction      │  Sentiment       │  Forecasting     │
│                  │                  │                  │                  │
│  🥇 Claude S4.5  │  🥇 GPT-5       │  🥇 Gemini Pro   │  🥇 Claude Opus  │
│  vs GPT-5        │  vs Gemini Pro   │  vs Claude Code  │  vs GPT-5        │
│                  │                  │                  │                  │
│  127 votes       │  89 votes        │  64 votes        │  103 votes       │
│  [View Battle]   │  [View Battle]   │  [View Battle]   │  [View Battle]   │
│                  │                  │                  │                  │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

## MVP Scope

### Phase 1: Proof of Concept (Weeks 1-4)

**Datasets: 3-5 curated examples**
- E-commerce sales (CSV, 10K rows)
- Customer churn (SQLite, 3 tables, 50K rows)
- Product reviews (JSONL, 5K records)
- Stock prices (CSV, 20K rows)
- Web analytics (CSV, 100K rows)

**Systems: 2-4 competitors**
- Claude 3.5 Sonnet (API)
- GPT-4 (API)
- Claude Code CLI (agent)
- One additional (Cursor or o1)

**Formats: 2-3 outputs**
- Jupyter Notebook (baseline)
- Streamlit (dashboard)
- Quarto/HTML (report)

**Voting: Simple comparison**
- Side-by-side view
- Binary choice (A vs B)
- Optional feedback tags
- No login required initially

**Success Metrics:**
- 1K votes in first month
- Clear winners in 70%+ of matchups
- <5 min end-to-end time
- <$1 cost per matchup

### Phase 2: Community Growth (Months 2-6)

**Add:**
- User-uploaded datasets (with moderation)
- More AI systems (5-10 total)
- More output formats (D3.js, Dash, CSV)
- User accounts & voting history
- Public dataset integration (Kaggle, HuggingFace)
- Automated code quality metrics

**Expand:**
- Larger datasets (up to 1M rows)
- More complex multi-table scenarios
- Domain-specific challenges
- Time-limited competitions

### Phase 3: Platform Maturity (Months 6-12)

**Features:**
- API access for researchers
- Custom model submissions
- White-label for enterprises
- Training data export
- Real-time leaderboards
- Community challenges & prizes

## Why This Matters

### For AI Companies
- Benchmark real analytical reasoning, not just coding
- Measure insight quality, not just syntax
- Human feedback on usefulness
- Competitive pressure drives improvement

### For Users (Data Analysts/Scientists)
- Discover which AI is best for their use case
- Learn from different analysis approaches
- See best practices from multiple systems
- Make informed tool choices
- Free? 

### For the Ecosystem
- Crowdsourced dataset of analysis patterns
- Training data for better analytical AI
- Format comparison insights
- Drive innovation in AI coding agents

### For Research
- Study human preferences in data analysis
- Compare LLM APIs vs. autonomous agents
- Evaluate communication vs. technical skills
- Understand format effectiveness

## Success Criteria

**MVP Success (3 months):**
- 40+ daily active voters
- 1K+ total votes collected
- 5+ AI systems compared
- Clear ranking differences emerge

## Next Steps

1. **Validate concept** - Show mockups to potential users
2. **Build kernel tool** - Jupyter execution environment
3. **Curate datasets** - 5 high-quality examples
5. **Automate pipeline** - Build backend infrastructure
6. **Launch MVP** - Public beta with 2 systems, 3 datasets
7. **Iterate** - Add systems and formats based on usage

## Open Questions

1. How to handle timeouts/failures gracefully?
2. Should we show code quality metrics alongside outputs?
3. How to prevent gaming (e.g., models optimizing for votes)?
4. Should voters see system names or blind comparison?
7. How to balance breadth (many formats) vs. depth (quality)?
8. Should we allow interactive refinement (user feedback loops)?

## Conclusion

Data Analysis Arena fills a critical gap in AI evaluation: measuring the ability to perform complete, useful data analysis - not just generate syntactically correct code.

By testing Python ability, communication, and visualization together, we evaluate what actually matters: can this AI system help people understand their data and make better decisions?

The crowdsourced voting approach ensures real human judgment of usefulness, not just automated metrics that can be gamed.

If successful, this becomes the standard benchmark for evaluating AI systems on data analysis and communication tasks - driving improvement across the industry and helping users choose the right tools for their needs.
