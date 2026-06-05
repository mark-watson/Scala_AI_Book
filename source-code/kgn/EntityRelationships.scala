// Copyright 2025-2026 Mark Watson. All rights reserved.

package kgn

object EntityRelationships:
  def results(entity1Uri: String, entity2Uri: String): QueryResult =
    val query =
      s"""|SELECT ?p WHERE {
          |  $entity1Uri ?p $entity2Uri .
          |  FILTER (!regex(str(?p), 'wikiPage', 'i'))
          |} LIMIT 10
          |""".stripMargin
    SparqlClient.query(query)
