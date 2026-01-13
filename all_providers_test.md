# Embedding Latency Benchmark Results

**Text analyzed:** 100 samples, avg 11.8 tokens each

## Key Finding

Embedding latency dominates RAG pipeline performance:
- Database reads: 8-20ms
- Embedding generation: 100-500ms (10-25x slower!)

## Results

| Provider/Model                |   Batch Size | P50 (ms)     | P95 (ms)     | P99 (ms)      |   Throughput (emb/s) |   Embeddings | Status   |
|:------------------------------|-------------:|:-------------|:-------------|:--------------|---------------------:|-------------:|:---------|
| Cohere/embed-v4.0             |            1 | 287.4 ±110.5 | 447.8 ±6.7   | 453.2 ±1.3    |                 32.1 |          100 | ✅ OK    |
| Cohere/embed-v4.0             |           10 | 909.6 ±49.7  | 954.5 ±4.8   | 958.4 ±1.0    |                 27.6 |          100 | ✅ OK    |
| Cohere/embed-v4.0             |           25 | 187.7 ±19.3  | 580.7 ±31.5  | 621.1 ±31.5   |                  3.9 |          100 | ✅ OK    |
| Gemini/gemini-embedding-001   |            1 | 334.9 ±282.4 | 634.1 ±12.4  | 644.1 ±2.5    |                 24.3 |          100 | ✅ OK    |
| Gemini/gemini-embedding-001   |           10 | 515.2 ±145.0 | 646.7 ±13.4  | 657.4 ±2.7    |                 48.9 |          100 | ✅ OK    |
| Gemini/gemini-embedding-001   |           25 | 305.5 ±21.0  | 482.0 ±103.0 | 625.7 ±453.7  |                  3.1 |          100 | ✅ OK    |
| Openai/text-embedding-3-large |            1 | 576.1 ±81.9  | 751.9 ±40.8  | 784.5 ±8.2    |                 17.4 |          100 | ✅ OK    |
| Openai/text-embedding-3-large |           10 | 607.0 ±41.4  | 646.2 ±2.2   | 647.9 ±0.4    |                 43.5 |          100 | ✅ OK    |
| Openai/text-embedding-3-large |           25 | 337.8 ±20.2  | 476.2 ±51.9  | 563.6 ±57.4   |                  2.9 |          100 | ✅ OK    |
| Openai/text-embedding-3-small |            1 | 986.3 ±31.9  | 1029.1 ±5.2  | 1033.3 ±1.0   |                 10.2 |          100 | ✅ OK    |
| Openai/text-embedding-3-small |           10 | 1032.0 ±69.6 | 1094.2 ±7.4  | 1100.2 ±1.5   |                 24.4 |          100 | ✅ OK    |
| Openai/text-embedding-3-small |           25 | 244.1 ±57.9  | 909.7 ±22.3  | 1133.2 ±793.4 |                  2.8 |          100 | ✅ OK    |
