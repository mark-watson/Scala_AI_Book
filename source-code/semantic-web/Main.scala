// Copyright 2025-2026 Mark Watson. All rights reserved.

package semantic_web

import java.io.File

@main def semanticWebDemo(): Unit =
  println("=" * 50)
  println("Semantic Web & SPARQL in Scala")
  println("=" * 50)

  // 1. Local Ontological Reasoning
  println("\n--- 1. Ontological Reasoning (Local RDF) ---")
  val model = JenaApis.createOntologyModel()
  val rdfFile = "data/news.n3"
  val resolvedRdfFile = if File(rdfFile).exists() then rdfFile else "semantic-web/" + rdfFile
  
  println(s"Loading RDF file from: $resolvedRdfFile")
  JenaApis.loadRdfFile(model, resolvedRdfFile)

  println("\nQuerying direct cities (kb:containsCity):")
  val directQuery = 
    """|PREFIX kb: <http://knowledgebooks.com/ontology#>
       |SELECT ?subject ?object WHERE { ?subject kb:containsCity ?object }
       |""".stripMargin
  val directResult = JenaApis.query(model, directQuery)
  println(directResult)

  println("\nQuerying inferred places (kb:containsPlace):")
  val inferredQuery = 
    """|PREFIX kb: <http://knowledgebooks.com/ontology#>
       |SELECT ?subject ?object WHERE { ?subject kb:containsPlace ?object }
       |""".stripMargin
  val inferredResult = JenaApis.query(model, inferredQuery)
  println(inferredResult)

  // 2. Remote SPARQL Query
  println("\n--- 2. Remote SPARQL Query (DBPedia) ---")
  val remoteQuery =
    """|SELECT ?p WHERE {
       |  <http://dbpedia.org/resource/Bill_Gates> ?p <http://dbpedia.org/resource/Microsoft> .
       |} LIMIT 10
       |""".stripMargin
  
  println("Executing remote SPARQL query on DBPedia endpoint...")
  try
    val remoteResult = JenaApis.queryRemote("https://dbpedia.org/sparql", remoteQuery)
    println(remoteResult)
  catch
    case e: Exception =>
      println(s"Failed to execute remote query: ${e.getMessage}")
      println("Please check your internet connection or the status of the DBPedia endpoint.")
