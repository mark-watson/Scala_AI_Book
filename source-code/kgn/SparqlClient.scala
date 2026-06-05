// Copyright 2025-2026 Mark Watson. All rights reserved.
//> using scala 3.6.4
//> using dep org.apache.jena:jena-arq:5.2.0

package kgn

import org.apache.jena.query.QueryExecution
import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters.*

case class QueryResult(variableList: List[String], rows: List[List[String]]):
  override def toString: String =
    val sb = StringBuilder(s"[QueryResult vars: $variableList\nRows:\n")
    for row <- rows do
      sb.append("  ").append(row).append("\n")
    sb.toString

object SparqlClient:
  private val ENDPOINT = "https://dbpedia.org/sparql"

  def query(sparqlQuery: String): QueryResult =
    val qexec = QueryExecution.service(ENDPOINT).query(sparqlQuery).build()
    try
      val results = qexec.execSelect()
      val vars = results.getResultVars.asScala.toList
      val rows = ListBuffer[List[String]]()
      while results.hasNext do
        val solution = results.nextSolution()
        val row = vars.map { v =>
          val node = solution.get(v)
          if node != null then node.toString else ""
        }
        rows.append(row)
      QueryResult(vars, rows.toList)
    finally
      qexec.close()
