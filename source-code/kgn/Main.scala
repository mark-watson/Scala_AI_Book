// Copyright 2025-2026 Mark Watson. All rights reserved.

package kgn

import scala.io.StdIn
import scala.util.Random

object Main:
  // ANSI terminal formatting codes
  val RESET = "\u001B[0m"
  val GREEN = "\u001B[32m"
  val YELLOW = "\u001B[33m"
  val PURPLE = "\u001B[35m"
  val CYAN = "\u001B[36m"

  private val DEMOS_LIST = List(
    "Bill Gates and Melinda Gates worked at Microsoft",
    "IBM opened an office in Canada",
    "Steve Jobs worked at Apple Computer and visited IBM and Microsoft in Seattle"
  )

  def main(args: Array[String]): Unit =
    println("=" * 50)
    println("Knowledge Graph Navigator (KGN) in Scala")
    println("=" * 50)
    println("Loading entity maps from files...")
    
    val allMaps = NerMaps.loadAll()
    println(s"Loaded ${allMaps.values.map(_.size).sum} entity mapping entries.")






    var running = true
    while running do
      println(s"\nEnter entities query (or press Enter for a random demo, 'exit'/'quit' to leave):")
      val rawInput = StdIn.readLine()
      if rawInput == null then
        running = false
      else
        val input = rawInput.trim
        if input.equalsIgnoreCase("exit") || input.equalsIgnoreCase("quit") then
          running = false
        else
          val query = if input.isEmpty then
            val selected = DEMOS_LIST(Random.nextInt(DEMOS_LIST.size))
            println(s"Running Demo: '$selected'")
            selected
          else
            input
          
          try
            processQuery(query, allMaps)
          catch
            case e: Exception =>
              println(s"Error processing query: ${e.getMessage}")
              e.printStackTrace()

  private def processQuery(query: String, allMaps: Map[String, Map[String, String]]): Unit =
    val kt = TextToDbpediaUris(query, allMaps)

    val people = kt.personNames.zip(kt.personUris).map(EntityAndDescription.apply).toList
    val companies = kt.companyNames.zip(kt.companyUris).map(EntityAndDescription.apply).toList
    val cities = kt.cityNames.zip(kt.cityUris).map(EntityAndDescription.apply).toList
    val countries = kt.countryNames.zip(kt.countryUris).map(EntityAndDescription.apply).toList

    println(s"\n${GREEN}Individual People:${RESET}")
    for person <- people do
      println(f"  ${GREEN}${person.entityName}%-25s${RESET} ${PURPLE}: ${person.entityUri}${RESET}")
      try println(EntityDetail.personAsString(person.entityUri))
      catch case e: Exception => println(s"  (Error fetching: ${e.getMessage})")

    println(s"\n${CYAN}Individual Companies:${RESET}")
    for company <- companies do
      println(f"  ${CYAN}${company.entityName}%-25s${RESET} ${YELLOW}: ${company.entityUri}${RESET}")
      try println(EntityDetail.companyAsString(company.entityUri))
      catch case e: Exception => println(s"  (Error fetching: ${e.getMessage})")

    println(s"\n${GREEN}Individual Cities:${RESET}")
    for city <- cities do
      println(f"  ${GREEN}${city.entityName}%-25s${RESET} ${PURPLE}: ${city.entityUri}${RESET}")
      try println(EntityDetail.cityAsString(city.entityUri))
      catch case e: Exception => println(s"  (Error fetching: ${e.getMessage})")

    println(s"\n${GREEN}Individual Countries:${RESET}")
    for country <- countries do
      println(f"  ${GREEN}${country.entityName}%-25s${RESET} ${PURPLE}: ${country.entityUri}${RESET}")
      try println(EntityDetail.countryAsString(country.entityUri))
      catch case e: Exception => println(s"  (Error fetching: ${e.getMessage})")

    // Check relationships between pairs of people
    for
      p1 <- people
      p2 <- people
      if p1 != p2
    do
      try
        val qr = EntityRelationships.results(p1.entityUri, p2.entityUri)
        if qr.rows.nonEmpty then
          println(s"\nRelationships between person '${p1.entityName}' and person '${p2.entityName}':")
          println(qr)
      catch
        case e: Exception => println(s"  (Error relationship: ${e.getMessage})")

    // Check relationships between people and companies
    for
      person <- people
      company <- companies
    do
      try
        val qr = EntityRelationships.results(person.entityUri, company.entityUri)
        if qr.rows.nonEmpty then
          println(s"\nRelationships between person '${person.entityName}' and company '${company.entityName}':")
          println(qr)
      catch
        case e: Exception => println(s"  (Error relationship: ${e.getMessage})")

    // Check relationships between pairs of companies
    for
      c1 <- companies
      c2 <- companies
      if c1 != c2
    do
      try
        val qr = EntityRelationships.results(c1.entityUri, c2.entityUri)
        if qr.rows.nonEmpty then
          println(s"\nRelationships between company '${c1.entityName}' and company '${c2.entityName}':")
          println(qr)
      catch
        case e: Exception => println(s"  (Error relationship: ${e.getMessage})")
