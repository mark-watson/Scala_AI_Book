//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package bravesearch

// Offline checks on the reply parser. Run with: scala-cli run . --main-class bravesearch.braveTest
@main def braveTest(): Unit =
  val sample = """{"web":{"results":[
    {"title":"T1","url":"https://a.example","description":"first hit"},
    {"title":"T2","url":"https://b.example","description":"second hit"},
    {"title":"T3","url":"https://c.example"}
  ]}}"""
  val hits = BraveSearch.parseReply(sample)
  assert(hits.size == 3, s"must parse 3 hits, got ${hits.size}")
  assert(hits.head == SearchHit("https://a.example", "T1", "first hit"), s"first hit wrong: ${hits.head}")
  assert(hits(2).snippet == "", "missing description defaults to empty")

  assert(BraveSearch.parseReply("""{"web":{}}""").isEmpty, "no results means empty list")
  assert(BraveSearch.parseReply("""{"other":1}""").isEmpty, "no web block means empty list")

  println("All Brave search tests passed.")
