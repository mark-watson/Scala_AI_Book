// Copyright 2025-2026 Mark Watson. All rights reserved.
//> using scala 3.6.4
//> using dep org.apache.jena:jena-arq:6.1.0

package semantic_web

import org.apache.jena.query.{QueryExecution, QueryFactory}
import org.apache.jena.rdf.model.{Model, ModelFactory}
import org.apache.jena.riot.{RDFDataMgr, RDFFormat}
import java.io.FileOutputStream
import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters.*

case class QueryResult(variableList: List[String], rows: List[List[String]]):
  override def toString: String =
    val sb = StringBuilder(s"[QueryResult vars: $variableList\nRows:\n")
    for row <- rows do
      sb.append("  ").append(row).append("\n")
    sb.toString

object JenaApis:
  def createOntologyModel(): Model =
    ModelFactory.createOntologyModel()

  def loadRdfFile(model: Model, filePath: String): Unit =
    model.read(filePath)

  def saveModelToTurtleFormat(model: Model, outputPath: String): Unit =
    val fos = FileOutputStream(outputPath)
    try RDFDataMgr.write(fos, model, RDFFormat.TURTLE)
    finally fos.close()

  def query(model: Model, sparqlQuery: String): QueryResult =
    val qexec = QueryExecution.model(model).query(sparqlQuery).build()
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

  def queryRemote(serviceUrl: String, sparqlQuery: String): QueryResult =
    val qexec = QueryExecution.service(serviceUrl).query(sparqlQuery).build()
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
