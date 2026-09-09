//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3
// Copyright 2026 Mark Watson. All rights reserved.

package bravesearch

// Brave web search client for agents and RAG. Port of
// brave_search/brave_search.lisp: GET the Brave endpoint with the
// subscription token, return (url, title, snippet) triples.
final case class SearchHit(url: String, title: String, snippet: String)

object BraveSearch:
  val endpoint = "https://api.search.brave.com/res/v1/web/search"

  def apiKey: String =
    sys.env.getOrElse("BRAVE_SEARCH_API_KEY",
      throw new IllegalStateException("Set BRAVE_SEARCH_API_KEY first."))

  // Parse the `web.results` array of a Brave JSON reply.
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

  def webSearch(query: String, count: Int = 5): List[SearchHit] =
    val response = requests.get(endpoint,
      params = Map("q" -> query, "count" -> count.toString),
      headers = Map("X-Subscription-Token" -> apiKey, "Accept" -> "application/json"),
      readTimeout = 60000)
    if response.statusCode != 200 then
      throw new java.io.IOException(s"Brave search failed (HTTP ${response.statusCode}).")
    parseReply(response.text())
