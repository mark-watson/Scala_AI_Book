# Semantic Web and SPARQL

The Semantic Web extends the World Wide Web by using standards to make data machine-readable. Rather than parsing unstructured HTML, computers can query data directly using semantic relationships.

In this chapter, we use the industry-standard **Apache Jena** framework in Scala 3 to read Resource Description Framework (RDF) data, execute SPARQL queries, and perform basic ontological reasoning using RDFS schemas.

All code is in `source-code/semantic-web`.

## From a Web of Documents to a Web of Data

Tim Berners-Lee, James Hendler, and Ora Lassila laid out the Semantic Web in a 2001 article. Their argument was simple: the web links documents written for people to read, but the meaning inside those documents is locked in prose that machines cannot use. If we instead publish **data** with explicit, shared meaning and link it across sites, software can combine facts from many sources and answer questions no single page states. This vision matured into today's **knowledge graphs**, the technology behind features like Google's info panels and the backbone of the next chapter's Knowledge Graph Navigator.

### The RDF Data Model

The foundation is the **Resource Description Framework (RDF)**, and its whole data model is one idea: every fact is a **triple** of subject, predicate, and object.

```$
(\text{subject}, \; \text{predicate}, \; \text{object})
```

A triple reads like a tiny sentence: *Bill Gates* (subject) *founded* (predicate) *Microsoft* (object). Collect many triples and they form a directed **graph**, where subjects and objects are nodes and predicates are the labeled edges between them. This is a fundamentally different shape from a relational database. A table fixes its columns in advance; an RDF graph grows by adding triples, so new kinds of facts need no schema change. To make facts globally mergeable, RDF names things with **URIs** (the same identifier scheme as web addresses). If two datasets both use `http://dbpedia.org/resource/Microsoft`, a machine knows they mean the same company, so graphs from different authors join automatically.

RDF is an abstract model, so it has several text **serializations**. The most verbose is RDF/XML; the most readable are N-Triples (one triple per line) and Turtle / N3, which let you group statements about the same subject. The data files in this chapter use N3, which is why a single subject can list many objects separated by semicolons and commas.

### SPARQL: Querying Graphs

If RDF is the data model, **SPARQL** is its query language, standardized by the W3C and roughly the SQL of the graph world. A SPARQL query is built from a **basic graph pattern**: a set of triples with some positions replaced by variables (written `?x`). The engine finds every way to bind those variables so the pattern matches triples in the graph, and returns the bindings. SPARQL has four query forms: `SELECT` returns a table of variable bindings, `ASK` returns true or false, `CONSTRUCT` builds a new RDF graph from the matches, and `DESCRIBE` returns the triples about a resource. Every query in this chapter is a `SELECT`.

## The Apache Jena Integration

Apache Jena is a comprehensive Java framework for building Semantic Web applications, and because it runs on the JVM we call it directly from Scala. Its central abstraction is the `Model`, an in-memory RDF graph you can load, query, and reason over. The SPARQL engine, called ARQ, executes queries against a model or against a remote endpoint. In **semantic-web/JenaApis.scala**, we wrap these APIs to load RDF files, execute local queries on models, and connect to remote SPARQL endpoints:

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

The key choice is `createOntologyModel`, which builds a model with an **RDFS reasoner** already attached, not a plain triple store. That reasoner is what makes the inference below work. The `query` method drives the standard ARQ result loop: build a `QueryExecution`, call `execSelect` to get a `ResultSet`, then iterate `QuerySolution` rows and read each variable's binding. We wrap each execution in a `try`/`finally` because a `QueryExecution` holds resources (and, for remote queries, a network connection) that must be closed. The Scala interop is smooth: `scala.jdk.CollectionConverters` bridges Jena's Java iterators to Scala collections with `asScala`.

## Local Ontological Reasoning and RDFS

One of the key strengths of RDF graphs is the ability to define schema constraints with **RDF Schema (RDFS)**, a lightweight ontology language. RDFS lets you state that one class is a subclass of another, or that one property is a **sub-property** of another, and it defines **entailment rules** that let a reasoner derive new triples the author never wrote down. This is the payoff of giving data explicit meaning: the machine can infer facts, not just retrieve them.

For example, we can define a rule stating that `containsCity` is a sub-property of `containsPlace`. In the Description Logic notation used for ontologies, the subsumption symbol `\sqsubseteq`$ reads "is more specific than":

```$
\mathit{containsCity} \sqsubseteq \mathit{containsPlace}
```

The relevant RDFS entailment rule (known as rdfs7) says: if `p`$ is a sub-property of `q`$, then every triple `(x, p, y)`$ also entails `(x, q, y)`$. So if the data contains the fact `(London, containsCity, Westminster)`, the reasoner automatically infers `(London, containsPlace, Westminster)` without that triple ever appearing in the file. We define this relationship structure in our data file **semantic-web/data/news.n3**:

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

For richer modeling, RDFS gives way to **OWL (the Web Ontology Language)**, which is grounded in Description Logics and can express constraints RDFS cannot, such as that two properties are inverses or that a class is defined by a restriction. Jena's ontology model supports OWL reasoning as well. One idea underlies all of it and is worth stating plainly: the Semantic Web uses the **Open World Assumption**. A fact missing from the graph is not therefore false, only unknown, unlike a SQL database, where absence means "does not exist". This assumption fits a web of data assembled from many incomplete sources.

## Local and Remote Queries

In **semantic-web/Main.scala**, we initialize a Jena ontology model, load the local data file, and perform two SPARQL queries that show inference at work:

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

The second query is the interesting one. Nowhere does the data assert a single `containsPlace` triple, yet the query for `containsPlace` returns the same cities as the query for `containsCity`. The reasoner supplies the missing triples on the fly from the sub-property rule. The two `WHERE` clauses are basic graph patterns with the subject and object left as variables, so each returns every subject-object pair connected by the given property.

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

This query fixes the subject and object and leaves the **predicate** as the variable `?p`, asking "in what ways is Bill Gates related to Microsoft?". It runs against **DBpedia**, a knowledge graph of structured facts extracted from Wikipedia and a central hub of the Linked Open Data cloud. Because DBpedia exposes a public SPARQL endpoint, our program queries billions of facts over HTTP without downloading anything, which is the Semantic Web promise made concrete: your code and someone else's data, joined by shared URIs.

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

The two local result sets are identical, which is the whole point: the inferred `containsPlace` query finds London and Paris even though the data only ever states `containsCity`, because the RDFS reasoner materialized the parent-property triples. The remote result shows the graph's answer to our question about Bill Gates and Microsoft: the predicates `founder` and `keyPerson`. We did not tell the program these facts; it discovered them by matching a pattern against a public knowledge graph. This same technique, resolving names to URIs and querying DBpedia for the relationships between them, is exactly what we build into a full application in the next chapter.
