//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3
// Copyright 2026 Mark Watson. All rights reserved.

package ragdocs

// Agentic RAG loop, after RAG/agents.lisp: rewrite the query into
// focused sub-queries, fan out retrieval across corpora, check the
// gathered context, and only then call the model.
final case class Corpus(name: String, chunks: List[Chunk])

object RagPipeline:
  // Break a compound question into 1-3 focused sub-queries.
  def rewriteQuery(question: String): List[String] =
    val parts = question.split("(?i)\\s+and\\s+|\\?").map(_.trim).filter(_.nonEmpty).toList
    (if parts.size <= 1 then List(question.trim) else parts).take(3)

  // Retrieve topK chunks per sub-query across all corpora, deduped.
  def gather(corpora: Seq[Corpus], question: String, topK: Int = 3): List[Chunk] =
    val all = corpora.flatMap(_.chunks)
    rewriteQuery(question)
      .flatMap(q => Retrieval.topK(q, all, topK))
      .distinct
      .take(topK * 2)

  def buildPrompt(question: String, context: Seq[Chunk]): String =
    val ctx = context.map(c => s"[${c.source}] ${c.text}").mkString("\n\n")
    s"""Answer the question using only the context below. Say "I do not know" when the context lacks the answer.

Context:
$ctx

Question: $question
Answer:"""

  // Full pass: gather context, build the prompt, and ask local Ollama
  // when it runs. Returns (answer, context) so callers can inspect both.
  def answer(corpora: Seq[Corpus], question: String, model: String = "mistral"): (String, List[Chunk]) =
    val context = gather(corpora, question)
    val prompt = buildPrompt(question, context)
    val answerText =
      try askOllama(prompt, model)
      catch case _: Exception => "[Ollama not running] Prompt built, no live answer. Start Ollama and retry."
    (answerText, context)

  private def askOllama(prompt: String, model: String): String =
    val payload = ujson.Obj("model" -> model, "prompt" -> prompt, "stream" -> false).render()
    val response = requests.post(
      "http://localhost:11434/api/generate",
      headers = Map("Content-Type" -> "application/json"),
      data = payload,
      readTimeout = 120000
    )
    ujson.read(response.text())("response").str
