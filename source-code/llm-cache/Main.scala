// Copyright 2026 Mark Watson. All rights reserved.

package llmcache

import java.nio.file.Files

@main def cacheDemo(): Unit =
  val db = Files.createTempFile("llm-cache-demo", ".db").toString
  val cache = CacheEngine(db)
  try
    cache.addCache("The quick brown fox jumps over the lazy dog")
    cache.addCache("Common Lisp is powerful")
    cache.addCache("Scala 3 has a clean syntax for AI work")
    println(s"Stored ${cache.countItems} items.")
    println(s"Lookup [fox]: ${cache.lookup(List("fox"))}")
    println(s"Lookup [scala, powerful] match-any: ${cache.lookup(List("scala", "powerful"), matchAny = true)}")
  finally cache.close()
