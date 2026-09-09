# Agentic RAG with Local Docs

A chat model alone can only quote what it saw in training. Retrieval-Augmented Generation (RAG) fixes that: fetch fresh text for each query, stuff it in the prompt, then let the model write from facts in hand. This chapter builds RAG in Scala 3 with no model downloads and no vector server. Plain TF-IDF vectors plus a BM25 term score rank the chunks, a rewrite step splits hard queries, and local Ollama writes the final answer when it runs.

All code is in `source-code/rag-docs`. Sample docs on solar power, EVs, and climate sit in `source-code/rag-docs/data`.

## Chunk and Rank with No Deps

Long docs do not fit a prompt, so `Chunking.split` cuts text at blank lines and slices long parts into overlapping word windows. Overlap keeps a fact split across a edge whole in at least one chunk. `Retrieval.rank` then scores each chunk two ways: cosine over TF-IDF weights, which rewards rare shared words, and BM25, which rewards dense term hits with length norm. The blend sorts best first:

```scala
val blended = 0.5 * cosine + 0.5 * bm25 / (bm25 + 10.0)
```

TF-IDF needs no training and runs in milliseconds. Swap in real embeds later if you must; the `Chunk` type and the `rank` shape stay put.

## Rewrite, Fan Out, Gather

Plain RAG runs one search per query. Agentic RAG plans first. `rewriteQuery` splits compound queries on "and" into up to three sharp sub-queries. `gather` runs each one, takes top chunks per part, and dedupes, so a query on solar plus EVs pulls both docs:

```scala
def gather(corpora: Seq[Corpus], question: String, topK: Int = 3): List[Chunk] =
  val all = corpora.flatMap(_.chunks)
  rewriteQuery(question)
    .flatMap(q => Retrieval.topK(q, all, topK))
    .distinct
    .take(topK * 2)
```

`buildPrompt` then frames the context with source tags and a firm order: use only this text, say "I do not know" when it falls short. That line cuts the most common RAG fault, fluent lies past the facts.

## Answer with Local Ollama

`answer` joins the parts: gather context, build the prompt, POST it to local Ollama, hand back both answer and sources. When Ollama is down it says so and still prints the prompt, so you can paste it to any hosted model by hand:

```scala
val (answerText, context) = RagPipeline.answer(corpora, question)
```

Run the demo and the checks:

```bash
cd source-code/rag-docs
scala-cli run . --main-class ragdocs.ragDemo
scala-cli run . --main-class ragdocs.ragTest
```

The tests need no model: chunk splits, solar queries rank the energy doc first, "and" queries split in two, gather spans both docs, and the prompt holds the sources. Grow this by adding the SQLite cache from the caching chapter in front of `askOllama`, so repeat queries cost zero.
