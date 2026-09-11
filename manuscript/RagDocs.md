# Agentic RAG with Local Docs

A chat model alone can only quote what it saw in training. Its weights freeze at a knowledge cutoff, so it cannot answer questions about your private files, a product that shipped last week, or this morning's news. When it does not know, it often invents a fluent answer anyway. Retrieval-Augmented Generation (RAG) fixes both problems. Instead of trusting the model's memory, RAG fetches text for each query, adds that text to the prompt, and lets the model write from facts in hand. The model still supplies the language; the facts come from a store you control and can update in seconds.

This chapter builds RAG in Scala 3 with no embedding model downloads and no vector server. Plain TF-IDF vectors plus a BM25 term score rank the chunks, a rewrite step splits hard queries, and local Ollama server uses a local LLM to write out the final answer for user queries.

All code is in `source-code/rag-docs`. Sample docs on solar power, EVs, and climate are in `source-code/rag-docs/data`.

## The RAG Pipeline

RAG has two phases. The **indexing** phase runs offline and turns a document collection into a searchable store:

1. **Load** each document as raw text.
2. **Chunk** the text into passages small enough to retrieve and to fit a prompt.
3. **Represent** each chunk as a vector of weights (here TF-IDF) or as an embedding.
4. **Store** the chunks and their vectors.

The **query** phase runs on every question:

1. **Rewrite** the question into one or more focused search queries.
2. **Retrieve** the chunks that best match each query.
3. **Augment** the prompt by adding the retrieved text.
4. **Generate** the answer with a language model that now sees the facts.

The word "augmented" names step three: you add the retrieved context to the prompt. Every other step exists to make that context relevant.

Chunking matters for two reasons. A prompt has a fixed context window, so a whole book will not fit. Retrieval also works best on short passages, because a single vector for a whole book mixes hundreds of topics and no query matches it well. Small passages give the ranker a narrow target and keep the prompt small.

RAG also gives you sources. Because each chunk carries its file name, the prompt labels where every fact came from, and the program prints the list. A reader can check the answer against the text. The prompt also tells the model to say "I do not know" when the context lacks the answer. That instruction removes the most common RAG fault, a fluent answer that drifts past the retrieved facts. It does not remove hallucination completely. A model can still misread or ignore context, and no prompt can recover a fact that retrieval never found. Retrieval quality matters most, which is why the ranking section below gets the most space.

## Lexical and Dense Retrieval

Two families of retrieval exist, and they fail in different ways.

**Lexical** retrieval matches words. It scores a chunk by the terms it shares with the query. TF-IDF and BM25 belong here. Lexical retrieval needs no training, runs in milliseconds, and handles rare tokens well: part numbers, names, and error codes. Its weakness is literal matching. A query about "automobiles" will not match a document that only says "cars", because the words differ.

**Dense** retrieval matches meaning. An embedding model maps the query and each chunk to a vector, and chunks near the query vector win. This captures synonyms and paraphrase, but it needs a model download and a vector store, and it can miss an exact rare token that a lexical match would catch.

**Hybrid** retrieval runs both and merges the scores, and it is the usual production choice. This chapter builds a hybrid in spirit without a model: BM25 supplies the keyword side, and TF-IDF cosine supplies a vector side, since TF-IDF weights are themselves vectors. The two scores blend into one ranking.

## Chunking Long Documents

`Chunking.split` breaks a document into paragraphs, then windows any paragraph that runs long:

```scala
object Chunking:
  def split(text: String, maxWords: Int = 120, overlap: Int = 20): List[String] =
    val paras = text.split("\n\\s*\n").map(_.replaceAll("\\s+", " ").trim).filter(_.nonEmpty)
    paras.flatMap(p => window(p, maxWords, overlap)).toList

  private def window(text: String, maxWords: Int, overlap: Int): List[String] =
    val words = text.split(" ").toList
    if words.size <= maxWords then List(text)
    else
      val step = math.max(maxWords - overlap, 1)
      words.sliding(maxWords, step).map(_.mkString(" ")).toList
```

The pattern `"\n\\s*\n"` matches a blank line: a newline, optional spaces, another newline. So the first split yields paragraphs. `replaceAll("\\s+", " ")` collapses runs of spaces, tabs, and newlines into a single space, and `filter(_.nonEmpty)` drops empty results. Each paragraph then goes through `window`. A short paragraph stays whole. A long one uses `sliding(maxWords, step)`, where `step = maxWords - overlap`. With the defaults, a window holds 120 words and advances 100, so 20 words repeat between neighbors.

Overlap is the important idea. A fact near the end of one window also appears at the start of the next, so a sentence split across a boundary stays whole in at least one chunk. The cost is duplication: overlap raises the number of chunks and the index size.

Paragraphs come first because a paragraph is a natural unit of topic. Cutting mid-paragraph first would separate a claim from its explanation. Only paragraphs longer than `maxWords` get windowed, so most chunks keep their boundaries.

The defaults trade precision against recall. Small chunks match a query more precisely but carry less context. Large chunks carry more context but weaken the match and spend more prompt tokens. The test pins this behavior: a 300-word string with `maxWords = 120` and `overlap = 20` yields 3 chunks, and the second window contains `word101` from the first.

## Ranking with TF-IDF and BM25

A retrieved item is a chunk plus its source file:

```scala
final case class Chunk(source: String, text: String)
```

The `source` travels with the text so the prompt can tag it and the demo can list it.

`Retrieval.tokenize` turns text into terms:

```scala
def tokenize(text: String): List[String] =
  text.toLowerCase.split("[^a-z0-9]+").filter(w => w.length > 2).toList
```

It lowercases, splits on anything that is not a letter or digit, and drops tokens of two characters or less. The split removes punctuation, and the length filter removes common short words such as "the", "of", and "is". This is a crude stopword filter. It also drops real short words such as "EV", a limit worth knowing.

`rank` scores every chunk against the query. It first builds the shared statistics:

```scala
def rank(query: String, chunks: Seq[Chunk]): List[(Chunk, Double)] =
  if chunks.isEmpty then return Nil
  val qTerms = tokenize(query).distinct
  if qTerms.isEmpty then return chunks.map(c => (c, 0.0)).toList
  val docs = chunks.map(c => tokenize(c.text))
  val n = docs.size.toDouble
  val df = qTerms.map(t => t -> docs.count(_.contains(t)).toDouble).toMap
  val idf = df.view.mapValues(d => math.log(1.0 + n / (d + 0.5))).toMap
  val avgLen = docs.map(_.size).sum.toDouble / n
```

- `qTerms` holds the distinct query tokens.
- `docs` holds the token list of every chunk.
- `n` is the number of chunks.
- `df`, the document frequency, counts how many chunks contain each query term.
- `idf`, the inverse document frequency, turns that count into a weight: `\mathrm{idf}(t) = \log(1 + N/(\mathrm{df}(t)+0.5))`$. A term in every chunk gets a low weight; a term in one chunk gets a high weight. The `0.5` and the `1 +` are smoothing, which keeps the weight positive and finite when a term is rare or absent.
- `avgLen` is the mean chunk length in tokens, which BM25 needs.

Then each chunk gets two scores and a blend:

```scala
  chunks.zip(docs).map { case (chunk, terms) =>
    val tf = terms.groupBy(identity).view.mapValues(_.size.toDouble).toMap
    val cosine = cosineScore(qTerms, tf, idf)
    val bm25 = qTerms.map { t =>
      val f = tf.getOrElse(t, 0.0)
      idf(t) * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * terms.size / avgLen))
    }.sum
    val blended = 0.5 * cosine + 0.5 * bm25 / (bm25 + 10.0)
    (chunk, blended)
  }.sortBy(-_._2).toList
```

`tf` is the term frequency of this chunk, a map from term to count. `cosine` and `bm25` score the chunk two ways, `blended` mixes them, and `sortBy(-_._2)` sorts best first.

TF-IDF cosine. The weight of a term in a chunk is `w(t,d) = \mathrm{tf}(t,d)\cdot\mathrm{idf}(t)`$: frequent in this chunk, rare in the collection. Both the query and the chunk become weight vectors, and cosine similarity measures the angle between them:

```$
\cos(q,d) = \frac{\sum_t w_q(t)\, w_d(t)}{\lVert w_q \rVert\, \lVert w_d \rVert}
```

Because the score divides by the vector lengths, a long chunk does not win just by holding more words. The code computes the query norm from the idf weights, the chunk norm from its TF-IDF weights, and the dot product over the query terms.

BM25 refines the term count with two controls. `k1 = 1.2` sets term frequency saturation: the first few hits add a lot, later hits add less, so a chunk that repeats a word fifty times cannot dominate the ranking. `b = 0.75` sets length normalization: long chunks contain more words by chance, so the score divides by the chunk length relative to the average. The full term contribution is:

```$
\mathrm{bm25}(q,d) = \sum_{t \in q} \mathrm{idf}(t)\cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1 - b + b\,\frac{|d|}{\mathrm{avgLen}}\right)}
```

Here `f(t,d)`$ is the term count in the chunk and `|d|`$ is the chunk length. As the count grows, the fraction approaches `k_1+1`$, so the score saturates. When the chunk is longer than average, the denominator grows and the score falls.

Cosine already lies in `[0, 1]`$, but BM25 is unbounded, so the two cannot be averaged directly. The code squashes BM25 with `\mathrm{bm25}/(\mathrm{bm25}+10)`$, which maps any positive score into `[0, 1)`$ and equals `0.5`$ at `\mathrm{bm25} = 10`$. The final score is:

```$
\mathrm{score}(q,d) = 0.5\cos(q,d) + 0.5\,\frac{\mathrm{bm25}(q,d)}{\mathrm{bm25}(q,d)+10}
```

Now both sides carry equal weight. Cosine rewards chunks that share rare words, and BM25 rewards chunks with dense term hits, adjusted for length. A blend of this kind is a cheap stand-in for the hybrid retrieval described earlier.

`topK` wraps `rank` and returns the chunk objects:

```scala
def topK(query: String, chunks: Seq[Chunk], k: Int): List[Chunk] =
  rank(query, chunks).take(k).map(_._1)
```

A worked example shows both the strength and the weakness of lexical ranking. On the sample docs, the demo question "How do solar panels help cut carbon emissions from home charging of electric cars?" ranks:

```text
electric-vehicles.txt    0.4035
climate-science.txt      0.3888
renewable-energy.txt     0.2931   (solar paragraph)
renewable-energy.txt     0.1117   (wind paragraph)
```

The question names solar panels, yet the electric-vehicle chunk wins. Lexical scoring counts every shared term. The question also says "home charging", "electric cars", and "carbon", and the EV text contains "home charger", "charging", "electric cars", and "carbon free". The solar chunk shares "solar", "home", "electric", "car", and "carbon", but fewer of the rarer terms. Lexical retrieval has no notion that "solar panels" is the subject. It measures word overlap, and the EV chunk overlaps more. The next section shows why the pipeline still answers well: it gathers more than the single top chunk.

## Rewriting and Fanning Out Queries

Plain RAG runs one search per question. Agentic RAG plans before it searches. The plan here is a query rewrite. `rewriteQuery` splits a compound question into focused sub-queries:

```scala
def rewriteQuery(question: String): List[String] =
  val parts = question.split("(?i)\\s+and\\s+|\\?").map(_.trim).filter(_.nonEmpty).toList
  (if parts.size <= 1 then List(question.trim) else parts).take(3)
```

The pattern splits on the word "and" or on a question mark, and `(?i)` makes the "and" match case-insensitive. Each part is trimmed and empties are dropped. A question with one part passes through whole. A compound question becomes up to three sub-queries. Splitting on "and" is crude, but it fits the common shape of a two-part question, and the cap of three keeps the search count bounded. The question mark also splits, so a multi-sentence question becomes several queries.

`gather` runs each sub-query and merges the results:

```scala
def gather(corpora: Seq[Corpus], question: String, topK: Int = 3): List[Chunk] =
  val all = corpora.flatMap(_.chunks)
  rewriteQuery(question)
    .flatMap(q => Retrieval.topK(q, all, topK))
    .distinct
    .take(topK * 2)
```

Every chunk from every corpus joins one pool, so a search can cross document lines. Each sub-query contributes its top `topK` chunks, `flatMap` flattens the lists, `distinct` removes chunks that two sub-queries both returned, and `take(topK * 2)` caps the prompt. With the defaults, the prompt holds at most six chunks.

This is where the EV result above stops being a problem. The single query still returns three chunks because `topK` is three, so the solar chunk reaches the model along with the EV and climate chunks. A compound query does more: "solar panels and electric cars" splits in two, each part retrieves its own top chunks, and the union covers both topics. That union is the agentic step. The program decides how many searches to run and how to combine them before it builds the prompt.

## Building the Grounded Prompt

`buildPrompt` frames the retrieved text and states the rules:

```scala
def buildPrompt(question: String, context: Seq[Chunk]): String =
  val ctx = context.map(c => s"[${c.source}] ${c.text}").mkString("\n\n")
  s"""Answer the question using only the context below. Say "I do not know" when the context lacks the answer.

Context:
$ctx

Question: $question
Answer:"""
```

Each chunk gets a bracketed source tag, and blank lines separate the chunks. The tag lets the model attribute a fact to a file and lets you verify it later. The instruction is strict on purpose: "using only the context" and "say I do not know". Without that constraint, a model falls back on its training weights and may answer from memory, which defeats the purpose of retrieval. The prompt ends with "Answer:" so the model continues from there.

The context is not free. Every chunk costs prompt tokens, and a long context raises cost and latency. Models also use facts at the start and end of a long context better than facts in the middle, so adding chunks past the useful ones can lower answer quality. The cap in `gather` and the rank order both keep the best chunks near the front.

## Answering with Local Ollama

`answer` runs the full pass:

```scala
def answer(corpora: Seq[Corpus], question: String, model: String = "mistral"): (String, List[Chunk]) =
  val context = gather(corpora, question)
  val prompt = buildPrompt(question, context)
  val answerText =
    try askOllama(prompt, model)
    catch case _: Exception => "[Ollama not running] Prompt built, no live answer. Start Ollama and retry."
  (answerText, context)
```

It gathers context, builds the prompt, and calls the model. On any failure it returns a fixed notice instead of throwing, and it always returns the context, so the caller can inspect which chunks were used. To see the prompt text, call `buildPrompt` directly; the demo prints the context sources.

`askOllama` sends one HTTP POST to the local server:

```scala
private def askOllama(prompt: String, model: String): String =
  val payload = ujson.Obj("model" -> model, "prompt" -> prompt, "stream" -> false).render()
  val response = requests.post(
    "http://localhost:11434/api/generate",
    headers = Map("Content-Type" -> "application/json"),
    data = payload,
    readTimeout = 120000
  )
  ujson.read(response.text())("response").str
```

The payload names the model, the prompt, and `stream = false`. With streaming off, the server buffers the whole answer and returns one JSON object, which is easy to parse in a batch program. The code reads the text from the `response` field. The two-minute read timeout covers the first call, when Ollama loads the model into memory before it generates a token. This is the same client shape as the Ollama chapter, reduced to the one call RAG needs.

## Loading the Docs and Running the Demo

`loadCorpora` reads every `.txt` file in a directory and turns it into a `Corpus`:

```scala
def loadCorpora(dir: String): List[Corpus] =
  val d = Paths.get(dir)
  if !Files.isDirectory(d) then return Nil
  Files.list(d).toArray.toList.collect {
    case p: java.nio.file.Path if p.toString.endsWith(".txt") =>
      val text = Files.readString(p)
      Corpus(p.getFileName.toString, Chunking.split(text).map(t => Chunk(p.getFileName.toString, t)))
  }
```

The file name becomes both the corpus name and each chunk's `source`, which is why the prompt tags and the demo output name the files. The demo ties the pieces together:

```scala
@main def ragDemo(): Unit =
  val corpora = loadCorpora("data")
  val total = corpora.map(_.chunks.size).sum
  println(s"Loaded ${corpora.size} corpora with $total total chunks.")
  val question = "How do solar panels help cut carbon emissions from home charging of electric cars?"
  println(s"\nQuestion: $question")
  println(s"Sub-queries: ${RagPipeline.rewriteQuery(question).mkString(" | ")}")
  val (answerText, context) = RagPipeline.answer(corpora, question)
  println(s"\nUsed ${context.size} context chunks from: ${context.map(_.source).distinct.mkString(", ")}")
  println(s"\n===== ANSWER =====\n$answerText")
```

Run it with:

```bash
cd source-code/rag-docs
scala-cli run . --main-class ragdocs.ragDemo
```

With Ollama off, the output shows the pipeline working and the answer step reporting the missing model:

```text
Loaded 3 corpora with 4 total chunks.

Question: How do solar panels help cut carbon emissions from home charging of electric cars?
Sub-queries: How do solar panels help cut carbon emissions from home charging of electric cars?

Used 3 context chunks from: electric-vehicles.txt, climate-science.txt, renewable-energy.txt

===== ANSWER =====
[Ollama not running] Prompt built, no live answer. Start Ollama and retry.
```

The demo question has no "and" and no question mark, so `rewriteQuery` returns it whole and the sub-query line repeats it. Retrieval still returns three chunks, one from each file, because `topK` is three. Start Ollama and pull the model to get a real answer:

```bash
ollama pull mistral
ollama serve
scala-cli run . --main-class ragdocs.ragDemo
```

## Offline Checks

`RagTest` pins each stage with plain `assert` calls, so the checks need no model and cost nothing:

```scala
// Chunking: short text stays whole, long text splits with overlap.
assert(Chunking.split("one two three").size == 1, "short text must stay whole")
val long = (1 to 300).map(i => s"word$i").mkString(" ")
val windows = Chunking.split(long, maxWords = 120, overlap = 20)
assert(windows.size == 3, s"300 words in 120-word windows with step 100 must give 3 chunks, got ${windows.size}")
assert(windows(1).contains("word101"), "windows must overlap")

// Retrieval: solar question ranks the energy chunk first.
val ranked = Retrieval.rank("solar panel electricity", corpora.head.chunks)
assert(ranked.head._1 == energy, s"energy chunk must rank first, got ${ranked.head._1.source}")
assert(Retrieval.rank("solar", Nil).isEmpty, "empty store ranks nothing")
assert(Retrieval.topK("battery range", corpora.head.chunks, 1) == List(cars), "topK must return the car chunk")

// Rewrite: compound questions split, simple ones pass through.
val subs = RagPipeline.rewriteQuery("How do solar panels work and how do EVs charge?")
assert(subs.size == 2, s"rewrite must split on 'and', got $subs")
assert(RagPipeline.rewriteQuery("What is solar?").size == 1, "simple question stays whole")

// Gather fans out across sub-queries and dedupes.
val gathered = RagPipeline.gather(corpora, "solar panels and electric cars", topK = 1)
assert(gathered.contains(energy) && gathered.contains(cars), s"gather must cover both parts, got ${gathered.map(_.source)}")

// Prompt carries the question and the sources.
val prompt = RagPipeline.buildPrompt("Q?", List(energy))
assert(prompt.contains("Q?") && prompt.contains("energy.txt"), "prompt must hold question and source")
```

The checks cover the whole chain: chunk counts and overlap, the ranking of the energy chunk for a solar query, the empty-store guard, `topK`, the "and" split, the pass-through for a simple question, the cross-document gather, and the prompt's question and source tags. Run them with:

```bash
scala-cli run . --main-class ragdocs.ragTest
```

A green run ends with:

```text
All RAG tests passed.
```

## Extending the Pipeline

The pipeline has clear extension points. Swap the ranker by replacing `Retrieval.rank` with an embedding model and a cosine score over dense vectors; the `Chunk` type and the `rank` signature stay. Swap the generator by pointing `askOllama` at a hosted API; the prompt does not change. Add a reranker after `gather` to score the top chunks with a cross-encoder and keep only the best few. Add the SQLite cache from the caching chapter in front of `askOllama` so repeat queries cost zero, and cache embeddings by chunk hash so reindexing is cheap. Add the Brave search client to pull live web snippets into the same pool when the local docs lack the answer.

The pieces in this chapter are the core of every RAG system: split the text, score the chunks, plan the searches, ground the prompt, and cite the sources. Production systems add scale, better embeddings, and reranking, but the shape stays the same.
