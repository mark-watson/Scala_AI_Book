# A SQLite Cache for LLM Calls

Hosted models bill per token and answer over the network, so every call costs money and time. Agents make this worse because they repeat themselves. The same FAQ lookup, the same document question, and the same web search run on every turn of a loop. A cache in front of the API turns a repeat call into a local disk read. This chapter builds that cache on SQLite in Scala 3, then explains where caching fits in an LLM stack, what a cache key must contain, and where a term match fails.

All code is in `source-code/llm-cache`. The example needs one dependency, declared in **project.scala**:

```scala
//> using scala 3.6.4
//> using dep org.xerial:sqlite-jdbc:3.53.4.0
```

## Why Cache Model Calls

Three types of costs can drastically decrease when a repeated call becomes a cache hit:

- **Money.** Providers charge per input and output token. A hit spends zero tokens.
- **Latency.** A local disk read takes a few milliseconds at most. A hosted call takes hundreds of milliseconds to many seconds.
- **Rate limits.** Providers cap requests per minute. A hit does not count against the cap.

A cache also lets a program run offline. If the answer sits on disk, the program runs without a network or a model server. That matters for tests, demos, and travel.

The cost is correctness. A cache returns an old answer, and an old answer can be wrong. Most of this chapter is about when a cached answer is safe to serve.

## Four Places to Cache

"Caching" names several different things in an LLM program. Keep them apart:

1. **Response cache.** Map a request to a stored reply. This chapter builds one.
2. **Prompt cache.** The provider stores the attention state of a repeated prefix and skips recomputing it. OpenAI, Anthropic, and Gemini all offer a form of this. It lowers the cost of the prefix tokens but still runs the model.
3. **Retrieval cache.** Store embeddings by chunk hash, or store search hits by query. Reindexing and re-searching are then cheap.
4. **Tool cache.** Store the result of a deterministic tool call, such as a currency conversion or a file read.

The response cache and the prompt cache are not alternatives. A prompt cache speeds up the calls you still make. A response cache removes calls you do not need to make at all.

## Exact Keys Versus Term Matching

A response cache needs a key. The strongest key is the full request. Serialize the request to JSON, hash it, and store the hash with the reply:

```
key = sha256(canonical_json(request))
```

A later request with the same key returns the same reply with no model call. This is an **exact-match cache**. It is safe because two identical requests should produce the same answer.

The example in this chapter takes a weaker route. It keys on a few words from the prompt and matches rows with SQL `LIKE`. This is a **term cache**, close to the bag-of-words retrieval the RAG chapter uses. A term cache can hit when the prompt differs, which is both its value and its risk. A prompt that shares words with a stored row can return an answer for a different question.

The reason to show the term cache is that it needs no hashing, no embedding model, and no schema for request parameters. It is a small, readable index, and it is the same shape as the Lisp `cache_engine` this example ports. Treat it as a starting point, not the final design. The last section shows the exact-match upgrade.

## What Belongs in a Cache Key

A reply depends on more than the prompt text. If any of these change, the cached reply may not apply:

- the model name and version,
- the system prompt,
- the message history,
- the temperature and top-p sampling settings,
- the maximum output tokens and stop sequences,
- the tool definitions and the tool choice,
- a random seed, when the provider supports one.

An exact-match cache should include all of them in the hashed request. The term cache in this chapter stores only the reply text, so it cannot check any of them. That is the main reason to keep its use narrow.

Sampling also limits caching. At temperature zero a model is close to deterministic, but providers can still change behavior between versions, and some hardware paths are not bit-exact. Treat a cached answer as a value that was correct once, and set a time to live that matches how fast the answer can change.

## Opening the Store

`CacheEngine` in **CacheEngine.scala** takes a database path, opens it through JDBC, and creates the table at construction:

```scala
class CacheEngine(dbPath: String) extends AutoCloseable:
  private val conn: Connection = DriverManager.getConnection(s"jdbc:sqlite:$dbPath")
  private def exec(sql: String): Unit =
    val st = conn.createStatement()
    try st.execute(sql) finally st.close()

  exec("CREATE TABLE IF NOT EXISTS cache (id INTEGER PRIMARY KEY, content TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
```

SQLite fits this job. It is one file, needs no server, and runs inside the process. The `jdbc:sqlite:` URL points at a path. A relative path resolves against the working directory, and `:memory:` gives a private in-memory store. The `CREATE TABLE IF NOT EXISTS` makes startup idempotent: the first run builds the table, and every later run finds it ready.

`id INTEGER PRIMARY KEY` is the rowid, so inserts are fast and each row has a stable number. `created_at` defaults to `CURRENT_TIMESTAMP`, which SQLite records in UTC. The purge later reads that column.

`CacheEngine` extends `AutoCloseable`, so a caller writes `try ... finally cache.close()`. The `exec` helper closes its statement in a `finally`, and every other method does the same, so an exception never leaks a cursor.

## Writing Rows Safely

`addCache` binds the text as a parameter:

```scala
def addCache(text: String): Unit =
  val ps = conn.prepareStatement("INSERT INTO cache (content) VALUES (?)")
  try
    ps.setString(1, text)
    ps.executeUpdate()
  finally ps.close()
```

The `?` placeholder is the important part. String concatenation would let a reply that contains a quote break the SQL, and it would allow SQL injection if any part of the text came from a user. Binding sends the text as data, never as code, so quotes, newlines, and semicolons in a model reply cannot change the statement. Use a prepared statement for every value, even when the source looks trusted.

## Looking Up by Terms

`lookup` builds one query per call. With no terms it lists rows. With terms it chains a `LIKE` filter per term, joined by `AND` or `OR`:

```scala
def lookup(terms: Seq[String], limit: Int = 3, matchAny: Boolean = false): List[String] =
  val base = if terms.isEmpty then "SELECT content FROM cache"
  else
    val joiner = if matchAny then " OR " else " AND "
    "SELECT content FROM cache WHERE " + terms.map(_ => "content LIKE ?").mkString(joiner)
  val ps = conn.prepareStatement(base + " LIMIT ?")
  try
    terms.zipWithIndex.foreach { case (t, i) => ps.setString(i + 1, s"%$t%") }
    ps.setInt(terms.size + 1, limit)
    val rs = ps.executeQuery()
    try Iterator.continually(rs).takeWhile(_.next()).map(_.getString(1)).toList
    finally rs.close()
  finally ps.close()
```

The shape of the query depends on the terms:

- **No terms** returns rows with no filter, capped by `limit`. This reads the store as a list.
- **`AND`** (the default) requires every term to appear in one row. Use it when the terms come from one prompt and must describe one reply.
- **`OR`** (`matchAny = true`) requires at least one term. Use it when any rare word from the prompt is enough to identify a useful row. This is the bag-of-words mode.

Each term binds as `%term%`, so `LIKE` matches the term anywhere in the text. `zipWithIndex` pairs each term with its placeholder number, and `setString(i + 1, ...)` fills placeholders starting at 1. The limit binds as an integer in the final placeholder slot. SQLite accepts a bound `LIMIT`, so no value is spliced into the SQL string.

The `LIMIT` matters because a caller usually drops the hits into a prompt, and prompt tokens cost money. Three hits is the default cap.

Three details of `LIKE` are worth knowing:

- A leading `%` disables any index, so the query scans the table. That is fine for thousands of rows and too slow for millions. A full-text index or an embedding index is the fix at scale.
- SQLite `LIKE` ignores case for ASCII letters, so `%fox%` also matches `Fox`. That is usually what you want for search.
- `%` and `_` are wildcards inside the term too. Binding stops SQL injection but does not stop a term that contains a wildcard from matching more than expected. If terms come from users, escape `%` and `_` and add an `ESCAPE` clause.

The method has no `ORDER BY`, so row order is not defined by the SQL standard. SQLite returns a table scan in rowid order today, which is insertion order. If you want newest first, add `ORDER BY id DESC` before the limit.

## Managing the Data Store

Four small methods manage the store:

```scala
def countItems: Long =
  val rs = conn.createStatement().executeQuery("SELECT COUNT(*) FROM cache")
  try { rs.next(); rs.getLong(1) } finally rs.close()

def clear(): Unit = exec("DELETE FROM cache")

def clearOlderThan(days: Int): Unit =
  exec(s"DELETE FROM cache WHERE created_at <= datetime('now', '${-days} days')")

def clearOlderThanOneWeek(): Unit = clearOlderThan(7)
```

`countItems` reads the row count, `clear` empties the table, and `clearOlderThan` deletes rows whose timestamp is at or before a cutoff. The cutoff is `datetime('now', ...)`, which returns UTC, and `created_at` is UTC too, so the comparison is consistent.

The modifier is `'${-days} days'`. For `days = 7` it becomes `'-7 days'`, a cutoff one week in the past. For `days = -1` it becomes `'1 days'`, a cutoff one day in the future, so every row is old enough and the table empties. That negative value is what the offline test uses to check the purge without waiting a week. The sign flip is easy to get wrong: a modifier such as `'--1 days'` is invalid, and SQLite returns `NULL` from `datetime`, which makes the comparison match nothing. The code avoids that by negating the value in Scala and letting the string carry one sign.

`clearOlderThanOneWeek` names the policy most apps want. A one week time to live drops stale facts, so a news query refreshes on its own.

The store has a time to live but no size cap and no least-recently-used eviction. The last section sketches both.

## Demo and Its Output

The demo in **Main.scala** writes three rows to a temp database and runs both lookup shapes:

```scala
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
```

It prints:

```
Stored 3 items.
Lookup [fox]: List(The quick brown fox jumps over the lazy dog)
Lookup [scala, powerful] match-any: List(Common Lisp is powerful, Scala 3 has a clean syntax for AI work)
```

The first lookup uses the default `AND` with one term, `fox`, so it matches the one row that holds it. The second uses `OR` with `scala` and `powerful`, so it matches both rows that hold either word. The `scala` row holds "Scala", and `LIKE` ignores ASCII case, so the lower-case term still matches.

The demo uses `Files.createTempFile`, so the database disappears with the OS temp directory and the demo leaves no state behind. `try ... finally cache.close()` releases the connection even if a call throws.

## Cache First, Model Second

In production pipelines we wire the data store cache in front of any model call. Take a few rare words from the prompt, look them up, and call the model only on a miss:

```scala
def cachedAnswer(cache: CacheEngine, prompt: String, terms: Seq[String])(
    callModel: String => String
): String =
  val hits = cache.lookup(terms, matchAny = true)
  if hits.nonEmpty then hits.head
  else
    val reply = callModel(prompt)
    cache.addCache(reply)
    reply
```

The `callModel` function is the only part that touches the network, so the same wrapper works for Ollama, OpenAI, or Gemini. The Ollama and model client chapters show how to fill it in. On a hit the wrapper returns the stored text and spends nothing. On a miss it calls the model, stores the reply, and returns it. Store only after a successful call, so a timeout does not cache an error string as an answer.

Three rules make this safe:

1. Pick terms that identify the question, not common words. `solar`, `payback`, and a project name separate questions; `how`, `does`, and `the` do not. The RAG chapter makes the same point about stopwords.
2. Match `AND` when the terms must describe one answer. Match `OR` only when any rare term is enough. The wrong choice is the main source of false hits.
3. Set a time to live that matches the answer. Facts that change weekly should not live a month.

This wrapper caches the reply, not the request, so the stored row does not record which prompt produced it. Two different prompts that share terms can collide, and the wrapper cannot tell. The next section addresses that.

## Safety, Staleness, and Cost

A cache is a correctness trade, so make the trade explicit.

**Staleness.** Every cached answer ages. A short time to live keeps facts fresh at the cost of more model calls. Choose the window from how fast the answer changes, not from a default.

**False hits.** A term match is not a semantic match. `lookup(List("cache"))` can return a row about CPU caches. When a wrong answer is expensive, store the original prompt with the reply and check that the query and the stored prompt are close, or move to an exact key or an embedding key.

**Cache poisoning.** Cached text often flows back into a prompt, exactly like retrieved context in the RAG chapter. If the text came from a web page or a user, it can carry instructions. Treat cached content as untrusted input and keep the same prompt rules you use for retrieval.

**Privacy.** The store can hold personal data, secrets, or proprietary text. A time to live, a `clear`, and a database file with tight permissions all matter. Never share one cache across users unless the key includes a user or tenant id.

**Concurrency.** SQLite allows one writer at a time. If two agent processes share the file, a write can fail with `SQLITE_BUSY`. Set a busy timeout so the driver waits and retries, and turn on write-ahead logging so readers do not block the writer:

```scala
val st = conn.createStatement()
st.execute("PRAGMA busy_timeout = 5000")
st.execute("PRAGMA journal_mode = WAL")
st.close()
```

**Cost.** The cache saves money when the savings exceed the work of maintaining it. The break-even is simple:

```
saved = hits x average_tokens x price_per_token
```

A cache costs a disk read per lookup and a small amount of storage. Both are near zero next to a token price, so even a low hit rate wins. Measure the hit rate, because a term scheme that misses often adds a query and stores rows for no benefit.

## Offline Checks

`cacheTest` in **CacheTest.scala** pins each behavior with plain `assert` calls, so the checks need no model and cost nothing:

```scala
assert(cache.countItems == 0, "fresh cache must be empty")
cache.addCache("The quick brown fox jumps over the lazy dog")
cache.addCache("Common Lisp is powerful")
assert(cache.countItems == 2, "count must be 2 after two adds")

assert(cache.lookup(List("fox")) == List("The quick brown fox jumps over the lazy dog"), "single term lookup")
assert(cache.lookup(List("quick", "dog")).size == 1, "AND lookup hits one row")
assert(cache.lookup(List("fox", "lisp")).isEmpty, "AND lookup across rows hits nothing")

assert(cache.lookup(List("fox", "lisp"), matchAny = true).size == 2, "OR lookup hits both rows")
assert(cache.lookup(List("fox", "lisp"), limit = 1, matchAny = true).size == 1, "limit trims rows")

assert(cache.lookup(Nil, limit = 5).size == 2, "empty terms list rows")
cache.clearOlderThan(-1)
assert(cache.countItems == 0, "purge with negative days clears all")
cache.addCache("back again")
cache.clear()
assert(cache.countItems == 0, "clear empties the cache")
```

Each assert names one rule:

- a fresh store is empty,
- `countItems` tracks the inserts,
- one term matches its row,
- `AND` matches only when both terms share a row, and matches nothing when they sit in different rows,
- `OR` matches both rows,
- the limit trims the result,
- empty terms list rows,
- a negative-day purge clears the table, and `clear` empties it.

Run the demo and the checks:

```bash
cd source-code/llm-cache
scala-cli run . --main-class llmcache.cacheDemo
scala-cli run . --main-class llmcache.cacheTest
```

A green run ends with:

```
All cache tests passed.
```

The first run pulls the SQLite driver with coursier. Later runs are offline.

## Extending the Cache

The store is small on purpose. Four changes turn it into a production cache.

**Exact keys.** Add a `key` column with a unique index and a `lookupExact(key)` method. Hash the canonical request to build the key. Keep the term index as a fallback when the exact key misses.

**Richer rows.** Store the prompt, the reply, the model name, the token counts, and a hit counter, not just the reply. Then the cache can check that a hit is really the same question, and it can report its own savings.

**Eviction.** A time to live handles age but not size. Add a cap and drop the oldest or least-used rows when the table grows past it. A `hits` column gives you least-used; an `accessed_at` column gives you least-recently-used.

**Embedding keys.** Replace `LIKE` with an embedding of the prompt and a nearest-neighbor search over stored vectors. This matches paraphrases that share no words. It needs an embedding model and a threshold, and a wrong threshold returns a confident wrong answer, so measure it against real queries.

These upgrades stack. The interface stays `lookup`, `addCache`, and a time to live, so the callers do not change.
