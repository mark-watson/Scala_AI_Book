# SQLite Cache for LLM Calls — Scala AI Book

A persistent prompt and reply cache on SQLite, ported from
`cache_engine` in the Loving Common Lisp repo. Check the cache by
key terms before paying for a repeat API call.

## Run

    scala-cli run . --main-class llmcache.cacheDemo

## Test (offline, uses a temp DB file)

    scala-cli run . --main-class llmcache.cacheTest

## Book Cover Material, Copyright, and License

This example code is released using the Apache 2 license.

Copyright 2026 Mark Watson. All rights reserved.
