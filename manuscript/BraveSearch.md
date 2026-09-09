# Brave Web Search for Agents and RAG

RAG needs fresh facts, and agents need live answers. This chapter adds a small Brave Search client in Scala 3 that returns URL, title, and snippet triples. It feeds the RAG chapter with web context and feeds the tools chapter with a search tool.

All code is in `source-code/brave-search`.

## One GET, Three Fields

Brave takes a GET with the query plus a token header and returns ranked web hits. `parseReply` reads the `web.results` array and pulls three fields per hit, defaulting a lost snippet to blank instead of throwing:

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

`webSearch` wraps that parse with the live GET. It reads the key from `BRAVE_SEARCH_API_KEY` and raises a plain error when the key is missing, so scripts fail fast with a fix to run. Get a key at https://brave.com/search/api.

## Wire It to RAG and Tools

A search client earns its keep when other parts call it. For RAG, save the top snippets as text files in the docs dir and the ranker picks them up with no code change. For the agent loop in the tools chapter, wrap `webSearch` as one more `ToolDefinition`:

```scala
ToolDefinition("web-search", "Search the web. Args: query.", List("query"),
  args => BraveSearch.webSearch(args.head).map(h => s"${h.title} ${h.url} ${h.snippet}").mkString("\n"))
```

That single line turns a file-bound agent into one that can check live facts.

Run the demo and the checks:

```bash
cd source-code/brave-search
BRAVE_SEARCH_API_KEY=your-key scala-cli run . --main-class bravesearch.braveDemo
scala-cli run . --main-class bravesearch.braveTest
```

The tests parse a canned reply offline, so they need no key and cost zero.
