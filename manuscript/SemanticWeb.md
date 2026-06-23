# Semantic Web and SPARQL

The Semantic Web extends the World Wide Web by using standards to make data machine-readable. Rather than parsing unstructured HTML, computers can query data directly using semantic relationships. 

In this chapter, we use the industry-standard **Apache Jena** framework in Scala 3 to read Resource Description Framework (RDF) data, execute SPARQL queries, and perform basic ontological reasoning using RDFS schemas.

All code is in `source-code/semantic-web`.

## The Apache Jena Integration

Apache Jena is a comprehensive Java framework for building Semantic Web applications. In **semantic-web/JenaApis.scala**, we wrap Jena's APIs to load RDF files, execute local queries on models, and connect to remote SPARQL endpoints:

```scala
//> using dep org.apache.jena:jena-arq:5.2.0

package semantic_web

import org.apache.jena.query.{QueryExecution, QueryFactory}
import org.apache.jena.rdf.model.{Model, ModelFactory}
import org.apache.jena.riot.{RDFDataMgr, RDFFormat}
import java.io.FileOutputStream
import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters.*

case class QueryResult(variableList: List[String], rows: List[List[String]])

object JenaApis:
  def createOntologyModel(): Model =
    ModelFactory.createOntologyModel()

  def loadRdfFile(model: Model, filePath: String): Unit =
    model.read(filePath)

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
    ...
```

## Local Ontological Reasoning and RDFS

One of the key strengths of RDF graphs is the ability to define schema constraints (RDFS) that allow the database engine to infer new facts that are not explicitly written.

For example, we can define a rule stating that `containsCity` is a sub-property of `containsPlace`:

{$$}
\text{containsCity} \sqsubseteq \text{containsPlace}
{/$$}

If the data contains the fact `(London, containsCity, Westminster)`, an ontological model will automatically infer the fact `(London, containsPlace, Westminster)`.

We define this relationships structure in our data file **semantic-web/data/news.n3**:

```turtle
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix kb:    <http://knowledgebooks.com/ontology#> .

# Schema definitions
kb:containsCity  rdfs:subPropertyOf  kb:containsPlace .

# Fact assertions
kb:newsItem_1  kb:containsCity  kb:London .
kb:newsItem_2  kb:containsCity  kb:Paris .
```

## Local and Remote Queries

In **semantic-web/Main.scala**, we initialize a Jena ontology model, load the local data file, and perform two SPARQL queries:

1. **Direct Query**: Searching for cities using the sub-property `kb:containsCity`.
2. **Inferred Query**: Searching for places using the parent property `kb:containsPlace`.

```scala
@main def semanticWebDemo(): Unit =
  // 1. Local Ontological Reasoning
  val model = JenaApis.createOntologyModel()
  val rdfFile = "data/news.n3"
  JenaApis.loadRdfFile(model, rdfFile)

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
```

Furthermore, we connect to the public **DBpedia SPARQL endpoint** to retrieve semantic links connecting Bill Gates to Microsoft:

```scala
  // 2. Remote SPARQL Query (DBpedia)
  val remoteQuery =
    """|SELECT ?p WHERE {
       |  <http://dbpedia.org/resource/Bill_Gates> ?p <http://dbpedia.org/resource/Microsoft> .
       |} LIMIT 10
       |""".stripMargin
  
  val remoteResult = JenaApis.queryRemote("https://dbpedia.org/sparql", remoteQuery)
  println(remoteResult)
```

## Running the Semantic Web Demo

Run the demo using:

```bash
scala-cli run .
```

This compiles the project, runs the local inferencing queries, and performs the remote HTTP SPARQL query against DBpedia:

```text
==================================================
Semantic Web & SPARQL in Scala
==================================================

--- 1. Ontological Reasoning (Local RDF) ---
Loading RDF file from: data/news.n3

Querying direct cities (kb:containsCity):
[QueryResult vars: [subject, object]
Rows:
  [http://knowledgebooks.com/ontology#newsItem_2, http://knowledgebooks.com/ontology#Paris]
  [http://knowledgebooks.com/ontology#newsItem_1, http://knowledgebooks.com/ontology#London]
]

Querying inferred places (kb:containsPlace):
[QueryResult vars: [subject, object]
Rows:
  [http://knowledgebooks.com/ontology#newsItem_2, http://knowledgebooks.com/ontology#Paris]
  [http://knowledgebooks.com/ontology#newsItem_1, http://knowledgebooks.com/ontology#London]
]

--- 2. Remote SPARQL Query (DBPedia) ---
Executing remote SPARQL query on DBPedia endpoint...
[QueryResult vars: [p]
Rows:
  [http://dbpedia.org/ontology/founder]
  [http://dbpedia.org/property/founder]
  [http://dbpedia.org/ontology/keyPerson]
]
```

As shown in the output, the inferred query successfully finds London and Paris as "places" due to the RDFS sub-property inference rule, and the DBpedia lookup identifies Bill Gates's role as a founder and key person of Microsoft.
