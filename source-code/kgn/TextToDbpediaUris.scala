// Copyright 2025-2026 Mark Watson. All rights reserved.

package kgn

import scala.io.Source
import java.io.File

case class EntityAndDescription(entityName: String, entityUri: String)

object NerMaps:
  private def textFileToMap(dataDir: String, fileName: String): Map[String, String] =
    val path = s"$dataDir$fileName"
    val resolvedPath = if File(path).exists() then path else "kgn/" + path
    val source = Source.fromFile(resolvedPath, "UTF-8")
    try
      val map = collection.mutable.Map[String, String]()
      for line <- source.getLines() do
        val tokens = line.split("\t")
        if tokens.length > 1 then
          val name = tokens(0).trim
          var uri = tokens(1).trim
          if !uri.startsWith("<") then uri = s"<$uri>"
          map(name) = uri
      map.toMap
    finally
      source.close()

  def loadAll(dataDir: String = "data/"): Map[String, Map[String, String]] =
    Map(
      "person" -> textFileToMap(dataDir, "PeopleDbPedia.txt"),
      "city" -> textFileToMap(dataDir, "CityNamesDbpedia.txt"),
      "company" -> textFileToMap(dataDir, "CompanyNamesDbPedia.txt"),
      "country" -> textFileToMap(dataDir, "CountryNamesDbpedia.txt"),
      "broadcastNetwork" -> textFileToMap(dataDir, "BroadcastNetworkNamesDbPedia.txt"),
      "musicGroup" -> textFileToMap(dataDir, "MusicGroupNamesDbPedia.txt"),
      "politicalParty" -> textFileToMap(dataDir, "PoliticalPartyNamesDbPedia.txt"),
      "tradeUnion" -> textFileToMap(dataDir, "TradeUnionNamesDbPedia.txt"),
      "university" -> textFileToMap(dataDir, "UniversityNamesDbPedia.txt")
    )

class TextToDbpediaUris(text: String, allMaps: Map[String, Map[String, String]]):
  val personUris = collection.mutable.ListBuffer[String]()
  val personNames = collection.mutable.ListBuffer[String]()
  val companyUris = collection.mutable.ListBuffer[String]()
  val companyNames = collection.mutable.ListBuffer[String]()
  val cityUris = collection.mutable.ListBuffer[String]()
  val cityNames = collection.mutable.ListBuffer[String]()
  val countryUris = collection.mutable.ListBuffer[String]()
  val countryNames = collection.mutable.ListBuffer[String]()

  private val categoryLists = Map(
    "person" -> (personNames, personUris),
    "company" -> (companyNames, companyUris),
    "city" -> (cityNames, cityUris),
    "country" -> (countryNames, countryUris)
  )

  // Tokenize the input text
  private val tokens = tokenize(text + " . . .")
  processText()

  private def tokenize(s: String): Array[String] =
    s.replaceAll("\\.", " . ")
     .replaceAll(",", " , ")
     .replaceAll("\\?", " ? ")
     .replaceAll("\n", " ")
     .replaceAll(";", " ; ")
     .replaceAll(" +", " ")
     .split(" ")

  private def processText(): Unit =
    var i = 0
    val size = tokens.length - 2
    while i < size do
      val n3gram = s"${tokens(i)} ${tokens(i + 1)} ${tokens(i + 2)}"
      val n2gram = s"${tokens(i)} ${tokens(i + 1)}"

      // Try 3-gram match (longest match first)
      val skip3 = tryMatch(n3gram, 3, i)
      if skip3 > 0 then
        i += skip3
      else
        // Try 2-gram match
        val skip2 = tryMatch(n2gram, 2, i)
        if skip2 > 0 then
          i += skip2
        else
          // Try 1-gram match
          val skip1 = tryMatch(tokens(i), 1, i)
          if skip1 > 0 then i += skip1
          else i += 1

  private def tryMatch(ngram: String, n: Int, startIndex: Int): Int =
    val checkOrder = List(
      "person", "city", "company", "country",
      "broadcastNetwork", "musicGroup", "politicalParty",
      "tradeUnion", "university"
    )
    var matchedSkip = 0
    var categories = checkOrder
    while categories.nonEmpty && matchedSkip == 0 do
      val catName = categories.head
      categories = categories.tail
      val map = allMaps(catName)
      map.get(ngram) match
        case Some(uri) =>
          println(s"$catName\t$startIndex\t${startIndex + n - 1}\t$ngram\t$uri")
          // If it's one of the core categories we track in public fields, add it
          categoryLists.get(catName) match
            case Some((names, uris)) =>
              if !uris.contains(uri) then
                names.append(ngram)
                uris.append(uri)
            case None => // other categories not tracked in public fields
          matchedSkip = n
        case None => // try next category
    matchedSkip

