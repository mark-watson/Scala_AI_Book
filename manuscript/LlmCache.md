# A SQLite Cache for LLM Calls

Hosted models bill per token, and agents repeat queries. A cache in front of the API pays for itself fast: store each prompt and reply in SQLite, check key terms first, call the model only on a miss. This chapter ports the Lisp cache engine to Scala 3 with the Xerial SQLite driver. The whole store is one class, one table, and seven small methods.

All code is in `source-code/llm-cache`.

## Opening the Store

`CacheEngine` takes a DB path and opens it with the stock JDBC call. No ORM, no pool. The schema runs at construction, so the first use builds the table and later uses find it ready:

```scala
class CacheEngine(dbPath: String) extends AutoCloseable:
  private val conn: Connection = DriverManager.getConnection(s"jdbc:sqlite:$dbPath")
  private def exec(sql: String): Unit =
    val st = conn.createStatement()
    try st.execute(sql) finally st.close()

  exec("CREATE TABLE IF NOT EXISTS cache (id INTEGER PRIMARY KEY, content TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
```

Each row holds a text plus a stamp. The stamp drives the purge later. `AutoCloseable` gives callers `try` plus `finally cache.close()`, and every statement closes in a `finally` so a throw never leaks a cursor.

## Storing Replies

`addCache` binds the text as a param, never splices it into SQL. That one habit keeps quotes and newlines in cached replies from breaking the insert:

```scala
def addCache(text: String): Unit =
  val ps = conn.prepareStatement("INSERT INTO cache (content) VALUES (?)")
  try
    ps.setString(1, text)
    ps.executeUpdate()
  finally ps.close()
```

## Finding: AND Versus OR

`lookup` builds one query per call. With no terms it lists rows. With terms it chains `LIKE` filters with `AND`, so every term must sit in one row. Pass `matchAny` and it chains with `OR`, a bag of words match across rows. The limit splices as an int, since SQLite takes no bind there:

```scala
def lookup(terms: Seq[String], limit: Int = 3, matchAny: Boolean = false): List[String] =
  val base = if terms.isEmpty then "SELECT content FROM cache"
  else
    val joiner = if matchAny then " OR " else " AND "
    "SELECT content FROM cache WHERE " + terms.map(_ => "content LIKE ?").mkString(joiner)
  val ps = conn.prepareStatement(base + s" LIMIT $limit")
  try
    terms.zipWithIndex.foreach { case (t, i) => ps.setString(i + 1, s"%$t%") }
    val rs = ps.executeQuery()
    try Iterator.continually(rs).takeWhile(_.next()).map(_.getString(1)).toList
    finally rs.close()
  finally ps.close()
```

Use `AND` when terms from one prompt must share a reply. Use `OR` when rare words from anywhere in the cache will do. Cap with `limit` since callers stuff hits into prompts, and prompts cost tokens.

## Tending the Store

Three small methods tend state. `countItems` reports size, `clear` wipes all, and `clearOlderThan` drops rows past an age in days. The one week helper names the policy most apps want:

```scala
def countItems: Long =
  val rs = conn.createStatement().executeQuery("SELECT COUNT(*) FROM cache")
  try { rs.next(); rs.getLong(1) } finally rs.close()

def clear(): Unit = exec("DELETE FROM cache")

def clearOlderThan(days: Int): Unit =
  exec(s"DELETE FROM cache WHERE created_at <= datetime('now', '-$days days')")

def clearOlderThanOneWeek(): Unit = clearOlderThan(7)
```

## Demo and Its Output

The demo in **llm-cache/Main.scala** writes three rows to a temp DB, then runs both lookup shapes:

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

The first lookup uses `AND` and hits one row. The second uses `OR` and spans two. The temp file vanishes with the OS temp dir, so the demo leaves no state behind.

## Cache First, Model Second

Use the store as a front door. Before each paid call, look up two or three rare words from the prompt. On hit, serve the stored text. On miss, call the model, print the reply, and store it. The one week purge drops stale facts so news queries refresh on their own:

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

The checks run on a temp DB and pin each shape: single term hit, `AND` across rows hitting nothing, `OR` hitting both rows, limit trimming to one, empty terms listing rows, and purge plus clear emptying the store. Green ends with:

```
All cache tests passed.
```

The first run pulls the SQLite driver with coursier; later runs are offline.
