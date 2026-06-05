// Copyright 2025-2026 Mark Watson. All rights reserved.

package kgn

object EntityDetail:

  def personAsString(entityUri: String): String =
    val query = personTemplate.format(entityUri, entityUri, entityUri, entityUri, entityUri)
    val qr = SparqlClient.query(query)
    qr.toString

  def companyAsString(entityUri: String): String =
    val query = companyTemplate.format(entityUri, entityUri, entityUri, entityUri, entityUri)
    val qr = SparqlClient.query(query)
    qr.toString

  def cityAsString(entityUri: String): String =
    val query = cityTemplate.format(entityUri, entityUri, entityUri, entityUri, entityUri)
    val qr = SparqlClient.query(query)
    qr.toString

  def countryAsString(entityUri: String): String =
    val query = countryTemplate.format(entityUri, entityUri, entityUri, entityUri, entityUri)
    val qr = SparqlClient.query(query)
    qr.toString

  private val companyTemplate =
    """|SELECT DISTINCT
       |    (GROUP_CONCAT (DISTINCT ?industry2; SEPARATOR=' | ') AS ?industry)
       |    (GROUP_CONCAT (DISTINCT ?netIncome2; SEPARATOR=' | ') AS ?netIncome)
       |    (GROUP_CONCAT (DISTINCT ?label2; SEPARATOR=' | ') AS ?label)
       |    (GROUP_CONCAT (DISTINCT ?comment2; SEPARATOR=' | ') AS ?comment)
       |    (GROUP_CONCAT (DISTINCT ?numberOfEmployees2; SEPARATOR=' | ') AS ?numberOfEmployees) {
       |  %s <http://www.w3.org/2000/01/rdf-schema#label> ?label2 .
       |            FILTER (lang(?label2) = 'en') .
       |  OPTIONAL { %s <http://www.w3.org/2000/01/rdf-schema#comment> ?comment2 . FILTER (lang(?comment2) = 'en') } .
       |  OPTIONAL { %s <http://dbpedia.org/ontology/industry> ?industry2 } .
       |  OPTIONAL { %s <http://dbpedia.org/ontology/netIncome> ?netIncome2 } .
       |  OPTIONAL { %s <http://dbpedia.org/ontology/numberOfEmployees> ?numberOfEmployees2 } .
       |} LIMIT 30""".stripMargin

  private val personTemplate =
    """|SELECT DISTINCT
       |    (GROUP_CONCAT (DISTINCT ?birthplace2; SEPARATOR=' | ') AS ?birthplace)
       |    (GROUP_CONCAT (DISTINCT ?label2; SEPARATOR=' | ') AS ?label)
       |    (GROUP_CONCAT (DISTINCT ?comment2; SEPARATOR=' | ') AS ?comment)
       |    (GROUP_CONCAT (DISTINCT ?almamater2; SEPARATOR=' | ') AS ?almamater)
       |    (GROUP_CONCAT (DISTINCT ?spouse2; SEPARATOR=' | ') AS ?spouse) {
       |  %s <http://www.w3.org/2000/01/rdf-schema#label> ?label2 .
       |            FILTER (lang(?label2) = 'en') .
       |  OPTIONAL { %s <http://www.w3.org/2000/01/rdf-schema#comment> ?comment2 . FILTER (lang(?comment2) = 'en') } .
       |  OPTIONAL { %s <http://dbpedia.org/ontology/birthPlace> ?birthplace2 } .
       |  OPTIONAL { %s <http://dbpedia.org/ontology/almaMater> ?almamater2 } .
       |  OPTIONAL { %s <http://dbpedia.org/ontology/spouse> ?spouse2 } .
       |} LIMIT 10""".stripMargin

  private val countryTemplate =
    """|SELECT DISTINCT
       |    (GROUP_CONCAT (DISTINCT ?areaTotal2; SEPARATOR=' | ') AS ?areaTotal)
       |    (GROUP_CONCAT (DISTINCT ?label2; SEPARATOR=' | ') AS ?label)
       |    (GROUP_CONCAT (DISTINCT ?comment2; SEPARATOR=' | ') AS ?comment)
       |    (GROUP_CONCAT (DISTINCT ?populationDensity2; SEPARATOR=' | ') AS ?populationDensity) {
       |  %s <http://www.w3.org/2000/01/rdf-schema#label> ?label2 .
       |                       FILTER (lang(?label2) = 'en') .
       |  OPTIONAL { %s <http://www.w3.org/2000/01/rdf-schema#comment> ?comment2 . FILTER (lang(?comment2) = 'en') } .
       |  OPTIONAL { %s <http://dbpedia.org/ontology/areaTotal> ?areaTotal2 } .
       |  OPTIONAL { %s <http://dbpedia.org/ontology/populationDensity> ?populationDensity2 } .
       |} LIMIT 30""".stripMargin

  private val cityTemplate =
    """|SELECT DISTINCT
       |    (GROUP_CONCAT (DISTINCT ?latitude_longitude2; SEPARATOR=' | ')
       |        AS ?latitude_longitude)
       |    (GROUP_CONCAT (DISTINCT ?populationDensity2; SEPARATOR=' | ') AS ?populationDensity)
       |    (GROUP_CONCAT (DISTINCT ?label2; SEPARATOR=' | ') AS ?label)
       |    (GROUP_CONCAT (DISTINCT ?comment2; SEPARATOR=' | ') AS ?comment)
       |    (GROUP_CONCAT (DISTINCT ?country2; SEPARATOR=' | ') AS ?country) {
       |  %s <http://www.w3.org/2000/01/rdf-schema#label> ?label2 .
       |             FILTER (lang(?label2) = 'en') .
       |  OPTIONAL { %s <http://www.w3.org/2000/01/rdf-schema#comment> ?comment2 . FILTER (lang(?comment2) = 'en') } .
       |  OPTIONAL { %s <http://www.w3.org/2003/01/geo/wgs84_pos#geometry> ?latitude_longitude2 } .
       |  OPTIONAL { %s <http://dbpedia.org/ontology/PopulatedPlace/populationDensity> ?populationDensity2 } .
       |  OPTIONAL { %s <http://dbpedia.org/ontology/country> ?country2 } .
       |} LIMIT 30""".stripMargin

