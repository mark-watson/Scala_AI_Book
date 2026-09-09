// Copyright 2026 Mark Watson. All rights reserved.

package llmcache

import java.nio.file.Files

// Offline checks on a temp DB file. Run with: scala-cli run . --main-class llmcache.cacheTest
@main def cacheTest(): Unit =
  val db = Files.createTempFile("llm-cache-test", ".db").toString
  val cache = CacheEngine(db)
  try
    assert(cache.countItems == 0, "fresh cache must be empty")
    cache.addCache("The quick brown fox jumps over the lazy dog")
    cache.addCache("Common Lisp is powerful")
    assert(cache.countItems == 2, "count must be 2 after two adds")

    // AND is the default: both terms must sit in one row.
    assert(cache.lookup(List("fox")) == List("The quick brown fox jumps over the lazy dog"), "single term lookup")
    assert(cache.lookup(List("quick", "dog")).size == 1, "AND lookup hits one row")
    assert(cache.lookup(List("fox", "lisp")).isEmpty, "AND lookup across rows hits nothing")

    // OR finds both rows; limit trims to one.
    assert(cache.lookup(List("fox", "lisp"), matchAny = true).size == 2, "OR lookup hits both rows")
    assert(cache.lookup(List("fox", "lisp"), limit = 1, matchAny = true).size == 1, "limit trims rows")

    // Empty terms list the cache; a negative-day purge clears all.
    assert(cache.lookup(Nil, limit = 5).size == 2, "empty terms list rows")
    cache.clearOlderThan(-1)
    assert(cache.countItems == 0, "purge with negative days clears all")
    cache.addCache("back again")
    cache.clear()
    assert(cache.countItems == 0, "clear empties the cache")
  finally cache.close()

  println("All cache tests passed.")
