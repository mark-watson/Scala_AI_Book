//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package bravesearch

@main def braveDemo(): Unit =
  try
    val hits = BraveSearch.webSearch("Scala 3 micro transformer from scratch")
    hits.foreach(h => println(s"- ${h.title}\n  ${h.url}\n  ${h.snippet.take(160)}\n"))
  catch case e: Exception => println(s"[skip live search] ${e.getMessage}")
