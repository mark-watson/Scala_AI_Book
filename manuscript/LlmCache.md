# A SQLite Cache for LLM Calls

Hosted models bill per token, and agents repeat queries. A cache in front of the API pays for itself fast: store each prompt and reply in SQLite, check key terms first, call the model only on a miss. This chapter ports the Lisp cache engine to Scala 3 with the Xerial SQLite driver.

All code is in `source-code/llm-cache`.

## Five Methods on One Table

The schema is one table with text plus a stamp. `addCache` stores a reply. `lookup` finds rows holding all terms, or any term with `matchAny`, capped by limit. `countItems`, `clear`, and `clearOlderThan` tend the store:

```scala
exec("CREATE TABLE IF NOT EXISTS cache (id INTEGER PRIMARY KEY, content TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")

def lookup(terms: Seq[String], limit: Int = 3, matchAny: Boolean = false): List[String]
```

LIKE with bound params keeps queries safe from quote tricks. The limit binds as a plain int in the SQL string, since SQLite takes no bind there.

## Cache First, Model Second

Use it as a front door. Before each paid call, look up two or three rare words from the prompt. On hit, serve the stored text. On miss, call the model, print the reply, and store it. The one week purge drops stale facts so news queries refresh on their own:

```scala
val hits = cache.lookup(List("solar", "payback"), matchAny = true)
val answer = if hits.nonEmpty then hits.head else fetchAndStore(prompt)
```

Pair this with the RAG chapter: cache final answers by query terms, and cache embeds by chunk hash. Pair it with the tools chapter: cache web search hits by query, since agents re-search the same facts each run.

Run the demo and the checks:

```bash
cd source-code/llm-cache
scala-cli run . --main-class llmcache.cacheDemo
scala-cli run . --main-class llmcache.cacheTest
```

Both use temp DB files, so they leave no state behind. The first run pulls the SQLite driver with coursier; later runs are offline.
