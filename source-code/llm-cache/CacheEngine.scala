// Copyright 2026 Mark Watson. All rights reserved.

package llmcache

import java.sql.{Connection, DriverManager}

// Persistent text cache on SQLite. Port of cache-engine/cache-engine.lisp:
// store model replies, then look them up by key terms before paying
// for another API call.
class CacheEngine(dbPath: String) extends AutoCloseable:
  private val conn: Connection = DriverManager.getConnection(s"jdbc:sqlite:$dbPath")
  private def exec(sql: String): Unit =
    val st = conn.createStatement()
    try st.execute(sql) finally st.close()

  exec("CREATE TABLE IF NOT EXISTS cache (id INTEGER PRIMARY KEY, content TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")

  def addCache(text: String): Unit =
    val ps = conn.prepareStatement("INSERT INTO cache (content) VALUES (?)")
    try
      ps.setString(1, text)
      ps.executeUpdate()
    finally ps.close()

  // Find up to `limit` rows holding the terms. matchAny picks OR
  // (bag of words) over the default AND.
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

  def countItems: Long =
    val rs = conn.createStatement().executeQuery("SELECT COUNT(*) FROM cache")
    try { rs.next(); rs.getLong(1) } finally rs.close()

  def clear(): Unit = exec("DELETE FROM cache")

  def clearOlderThan(days: Int): Unit =
    exec(s"DELETE FROM cache WHERE created_at <= datetime('now', '-$days days')")

  def clearOlderThanOneWeek(): Unit = clearOlderThan(7)

  def close(): Unit = conn.close()
