//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package ragdocs

// Retrieval over local chunks: TF-IDF cosine plus a BM25 term score.
// Mirrors the autocontext idea (keyword rank plus vector rank) with
// no model downloads: the vectors are plain TF-IDF term weights.
final case class Chunk(source: String, text: String)

object Retrieval:
  def tokenize(text: String): List[String] =
    text.toLowerCase.split("[^a-z0-9]+").filter(w => w.length > 2).toList

  private val k1 = 1.2
  private val b = 0.75

  // Rank chunks for a query. Returns (chunk, score) sorted best first.
  def rank(query: String, chunks: Seq[Chunk]): List[(Chunk, Double)] =
    if chunks.isEmpty then return Nil
    val qTerms = tokenize(query).distinct
    if qTerms.isEmpty then return chunks.map(c => (c, 0.0)).toList
    val docs = chunks.map(c => tokenize(c.text))
    val n = docs.size.toDouble
    val df = qTerms.map(t => t -> docs.count(_.contains(t)).toDouble).toMap
    val idf = df.view.mapValues(d => math.log(1.0 + n / (d + 0.5))).toMap
    val avgLen = docs.map(_.size).sum.toDouble / n
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

  private def cosineScore(qTerms: List[String], tf: Map[String, Double], idf: Map[String, Double]): Double =
    val qNorm = math.sqrt(qTerms.map(t => idf.getOrElse(t, 0.0) * idf.getOrElse(t, 0.0)).sum)
    val dNorm = math.sqrt(tf.map { case (t, f) => val w = f * idf.getOrElse(t, 0.0); w * w }.sum)
    if qNorm == 0.0 || dNorm == 0.0 then 0.0
    else qTerms.map(t => idf.getOrElse(t, 0.0) * tf.getOrElse(t, 0.0) * idf.getOrElse(t, 0.0)).sum / (qNorm * dNorm)

  def topK(query: String, chunks: Seq[Chunk], k: Int): List[Chunk] =
    rank(query, chunks).take(k).map(_._1)
