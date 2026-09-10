# Brave Web Search for Agents and RAG

RAG needs fresh facts, and agents need live answers. A model's weights freeze at training time, so anything after its cutoff or behind a login is invisible to it. A search API closes that gap. This chapter adds a small Brave Search client in Scala 3 that returns URL, title, and snippet triples. It feeds the RAG chapter with web context and feeds the tools chapter with a search tool.

Brave runs its own crawl and index rather than reselling another engine's results, so the API returns the same independent index the Brave browser uses. That matters for a book example: the reply shape is documented and stable, and the free allowance is enough to run the demos here.

All code is in `source-code/brave-search`.

## Getting a Key and the Request Shape

Sign up at https://brave.com/search/api and create a key in the dashboard. The pricing page lists web search at $5 per 1,000 requests, and every account gets a monthly credit that covers the first requests free, so development and the demos in this chapter cost nothing. Check that page for the current allowance before you ship.

Every call is one GET to a fixed endpoint with the query in the URL and the key in a header:

```
GET https://api.search.brave.com/res/v1/web/search?q=scala+3&count=5
X-Subscription-Token: YOUR_API_KEY
Accept: application/json
```

Three details matter. The key travels in `X-Subscription-Token`, not as a Bearer token or an OAuth flow; a client that sends the usual `Authorization: Bearer` header gets a 401. The path carries a version segment, `res/v1`, so a future breaking change can land under a new version without moving the old one. And `q` is the only required parameter.

The query string is a small API of its own. The parameters an agent or RAG client reaches for most:

| Parameter | Meaning |
| --- | --- |
| `q` | The query. Required, at most 600 characters or 75 words. |
| `count` | Web results to return, 1 to 20. Default 20. |
| `offset` | Page number, 0 to 9. Combine with `count` to page. |
| `country` | Two-letter country code, default `US`. |
| `search_lang` | Language of results, default `en`. |
| `freshness` | Age filter: `pd` (day), `pw` (week), `pm` (month), `py` (year), or `YYYY-MM-DDtoYYYY-MM-DD`. |
| `safesearch` | `off`, `moderate` (default), or `strict`. |
| `text_decorations` | Highlight markers in snippets, default `true`. |
| `extra_snippets` | Up to five additional excerpts per result. |

`freshness` is the one to reach for when a question is about "now": an agent asked about today's news should send `freshness=pd` rather than trust the ranking to prefer recent pages.

## Reading the Reply

Brave returns one JSON object. Its top level has a `type`, a `query` block that echoes and normalizes the request, and one block per result family the plan exposes: `web`, `news`, `videos`, `discussions`, `faq`, `infobox`, `locations`, and a `mixed` block that gives the preferred display order across them. A client that only wants web pages can ignore everything but `web`.

```
{
  "type": "search",
  "query": { "original": "scala 3", "more_results_available": true },
  "web": {
    "type": "search",
    "results": [
      {
        "title": "Scala 3",
        "url": "https://example.com/scala3",
        "description": "Scala 3 is <strong>the</strong> next ...",
        "page_age": "2025-01-01T00:00:00",
        "language": "en",
        "extra_snippets": ["...", "..."]
      }
    ]
  }
}
```

Each entry in `web.results` carries many fields, but a retrieval client needs three: `url`, `title`, and `description`. The `description` is the snippet. It arrives with `<strong>` tags around the query terms because `text_decorations` defaults to `true`; a program that renders the snippet as HTML wants them, but one that feeds the text to a model should strip them first. The `query` block repays a look for two fields: `altered` holds the spell-corrected query, and `more_results_available` says whether another page exists, so you can page with `offset` without guessing.

## The Client

The client is one file, `BraveSearch.scala`. Two lihaoyi libraries do the work: `requests` for HTTP and `ujson` for JSON. The build directives sit at the top:

```scala
//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3

package bravesearch

final case class SearchHit(url: String, title: String, snippet: String)

object BraveSearch:
  val endpoint = "https://api.search.brave.com/res/v1/web/search"
```

`SearchHit` is the only data type. A case class gives `equals` and `toString` for free, which the offline test leans on. Keeping it to three fields means the RAG and tool code never learns about Brave's larger reply; if the API grows a field the client wants, it lands here and nowhere else.

The key comes from the environment, never from source:

```scala
  def apiKey: String =
    sys.env.getOrElse("BRAVE_SEARCH_API_KEY",
      throw new IllegalStateException("Set BRAVE_SEARCH_API_KEY first."))
```

A missing key fails before any socket opens, and the message names the variable to set. Hardcoding a key in a book example is the fastest way to leak it, so the demo and the tests both run without one.

## Parsing the Reply

`parseReply` turns the JSON string into a list of hits and never throws on a missing field:

```scala
  def parseReply(json: String): List[SearchHit] =
    val data = ujson.read(json)
    data.obj.get("web").flatMap(_.obj.get("results")) match
      case None => Nil
      case Some(results) =>
        results.arr.toList.map { r =>
          SearchHit(
            url = r.obj.get("url").map(_.str).getOrElse(""),
            title = r.obj.get("title").map(_.str).getOrElse(""),
            snippet = r.obj.get("description").map(_.str).getOrElse("")
          )
        }
```

The chain reads as a set of nested optional lookups. `ujson.read` parses the text into a `ujson.Value`, and `.obj` views it as a map. `.get("web")` returns `Option[ujson.Value]`, and `flatMap(_.obj.get("results"))` walks one level deeper only if `web` exists. If either is absent the whole expression is `None`, and the match returns an empty list. That covers the two shapes Brave sends for a query with no web hits: a reply with no `web` block, and a `web` block with no `results` array.

For each result, `.get("url").map(_.str).getOrElse("")` reads a field and falls back to the empty string. This is deliberate. Brave marks `description` optional, so a hit without one is normal, and a client that threw on it would drop the whole reply over one missing snippet. A blank snippet is harmless downstream; a crash is not. The same three lines handle `title` and `url`, which the schema calls required but which get the same tolerance so one odd result cannot sink the batch.

## Running a Search

`webSearch` wraps the parse with the live GET:

```scala
  def webSearch(query: String, count: Int = 5): List[SearchHit] =
    val response = requests.get(endpoint,
      params = Map("q" -> query, "count" -> count.toString),
      headers = Map("X-Subscription-Token" -> apiKey, "Accept" -> "application/json"),
      readTimeout = 60000)
    if response.statusCode != 200 then
      throw new java.io.IOException(s"Brave search failed (HTTP ${response.statusCode}).")
    parseReply(response.text())
```

`requests.get` takes the params as a map, so it URL-encodes the query for you: a space or an ampersand in the query cannot break the request. The two headers are the subscription token and the JSON accept type. `readTimeout = 60000` is 60 seconds in milliseconds, generous for a search call but enough to stop a batch job from hanging forever on a stalled connection.

The status check comes before parsing on purpose. A non-200 body is an error page or a JSON error object, not a result set, and feeding it to `parseReply` would yield a misleading empty list. Throwing an `IOException` with the status code keeps the failure visible.

The default `count = 5` is a book choice, not an API limit. The API allows up to 20, but five snippets are usually enough for a RAG prompt, and a smaller count keeps the prompt short and the response cheap. Raise it when a task needs broader coverage.

## Wiring It to RAG and Tools

A search client earns its keep when other parts call it. For RAG, save the top snippets as text files in the docs dir and the ranker picks them up with no code change. For the agent loop in the tools chapter, wrap `webSearch` as one more `ToolDefinition`:

```scala
ToolDefinition("web-search", "Search the web. Args: query.", List("query"),
  args => BraveSearch.webSearch(args.head).map(h => s"${h.title} ${h.url} ${h.snippet}").mkString("\n"))
```

That single line turns a file-bound agent into one that can check live facts. The tool takes one argument, the query, and returns one line per hit with the title, URL, and snippet joined by spaces. The loop's text protocol reads that block like any other tool output, so the model can cite the URL and quote the snippet in its next step. The RAG path is the same data with a different consumer: the snippets become extra chunks beside the local docs, and the ranker treats web text and file text alike.

## Demo and Checks

The demo prints the title, URL, and the first 160 characters of each snippet:

```scala
@main def braveDemo(): Unit =
  try
    val hits = BraveSearch.webSearch("Scala 3 micro transformer from scratch")
    hits.foreach(h => println(s"- ${h.title}\n  ${h.url}\n  ${h.snippet.take(160)}\n"))
  catch case e: Exception => println(s"[skip live search] ${e.getMessage}")
```

The `catch` is what lets the demo run in a room with no key. With a key set, it prints real hits:

```
- "From Scratch" Series 3: Micro-Transformers | by Aranya Ray | Medium
  https://medium.com/@aranya.ray1998/from-scratch-series-3-micro-transformers-ea3347012afb
  Micro-Transformers is <strong>a minimalist approach towards building transformer architectures from scratch</strong> ...
```

Note the `<strong>` tags in the snippet. They come straight from Brave's highlighting, which is another reason to clean the text before it reaches a model.

The tests parse a canned reply offline, so they need no key and cost zero:

```scala
val sample = """{"web":{"results":[
  {"title":"T1","url":"https://a.example","description":"first hit"},
  {"title":"T2","url":"https://b.example","description":"second hit"},
  {"title":"T3","url":"https://c.example"}
]}}"""
val hits = BraveSearch.parseReply(sample)
assert(hits.size == 3, ...)
assert(hits.head == SearchHit("https://a.example", "T1", "first hit"), ...)
assert(hits(2).snippet == "", "missing description defaults to empty")
assert(BraveSearch.parseReply("""{"web":{}}""").isEmpty, ...)
assert(BraveSearch.parseReply("""{"other":1}""").isEmpty, ...)
```

The sample copies the real reply shape down to the nesting, and the third hit omits `description` to prove the default. The last two checks pin the empty cases: a `web` block with no `results`, and a reply with no `web` block at all. Since the parser is a pure function from string to list, the whole suite runs without a network and without a key.

Run the demo and the checks:

```bash
cd source-code/brave-search
BRAVE_SEARCH_API_KEY=your-key scala-cli run . --main-class bravesearch.braveDemo
scala-cli run . --main-class bravesearch.braveTest
```

## Extending the Client

The client is deliberately narrow, and the API leaves room to grow it.

- **Clean the snippets.** Strip `<strong>` and other tags, or send `text_decorations=false` to get plain text from the server.
- **Filter by time.** Add a `freshness` parameter and pass `pd` or `pw` for news questions.
- **Get more text.** Set `extra_snippets=true` and merge the extra excerpts into the snippet; each result carries up to five more.
- **Page results.** When `query.more_results_available` is true, send a larger `offset` to fetch the next page.
- **Read other verticals.** The same reply carries `news`, `videos`, and `discussions` blocks; parse them the way `parseReply` reads `web`.
- **Try the agent endpoint.** Brave also exposes `/res/v1/llm/context`, a variant tuned for models that returns longer, pre-cleaned passages. The request shape is the same; only the parser changes.

Those changes all live behind `SearchHit`, so the RAG pipeline and the tool registry keep their current contract while the client grows.
