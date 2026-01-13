# Embedding Latency Benchmark Results

**Text analyzed:** 25 samples, avg 14.1 tokens each

## Key Finding

Embedding latency dominates RAG pipeline performance:
- Database reads: 8-20ms
- Embedding generation: 100-500ms (10-25x slower!)

## Results

| Provider/Model                |   Batch Size |   P50 (ms) |   P95 (ms) |   P99 (ms) |   Throughput (emb/s) |   Embeddings | Status   |
|:------------------------------|-------------:|-----------:|-----------:|-----------:|---------------------:|-------------:|:---------|
| Openai/text-embedding-3-large |            1 |      247.8 |      315   |      329.4 |                  7.5 |           25 | ✅ OK    |
| Openai/text-embedding-3-large |            2 |      312.8 |      940.5 |     1042.6 |                  4.5 |           25 | ✅ OK    |
| Openai/text-embedding-3-small |            1 |      390.4 |      689   |      751.4 |                  2.5 |           25 | ✅ OK    |
| Openai/text-embedding-3-small |            2 |      225.5 |      554.8 |      589.5 |                  3.5 |           25 | ✅ OK    |
