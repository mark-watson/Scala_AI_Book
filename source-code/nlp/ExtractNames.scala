// Copyright 2025-2026 Mark Watson. All rights reserved.

package nlp

import scala.io.Source
import java.io.File

/** A helper collection to collect unique names and keep count of their occurrences. */
class ScoredList(val maxToKeep: Int = 999999):
  private val counts = collection.mutable.Map[String, Int]().withDefaultValue(0)

  def addValue(text: String): Unit =
    counts(text) = counts(text) + 1
    if counts.size > maxToKeep then
      val minKey = counts.minBy(_._2)._1
      counts.remove(minKey)

  def getValuesAsString: String =
    counts.toList
      .sortBy(-_._2)
      .map((text, score) => s"$text:$score")
      .mkString(", ")

class ExtractNames(dataDir: String = "data/"):
  private val firstNameSet = collection.mutable.Set[String]()
  private val lastNameSet = collection.mutable.Set[String]()
  private val prefixSet = collection.mutable.Set[String]()
  private val placeNameMap = collection.mutable.Map[String, String]()

  // Resolve directory path
  private val resolvedDir =
    if File(dataDir).exists() then dataDir
    else if File("nlp/" + dataDir).exists() then "nlp/" + dataDir
    else throw java.io.FileNotFoundException(s"Data directory not found at $dataDir or nlp/$dataDir")

  // Load sets from files
  loadSet(resolvedDir + "firstnames.txt", firstNameSet)
  loadSet(resolvedDir + "lastnames.txt", lastNameSet)
  loadSet(resolvedDir + "prefixnames.txt", prefixSet)
  loadPlaceMap(resolvedDir + "placenames.txt", placeNameMap)

  private def loadSet(filePath: String, set: collection.mutable.Set[String]): Unit =
    val source = Source.fromFile(filePath)
    try
      for line <- source.getLines() do
        val trimmed = line.trim
        if trimmed.nonEmpty then set.add(trimmed)
    finally
      source.close()

  private def loadPlaceMap(filePath: String, map: collection.mutable.Map[String, String]): Unit =
    val source = Source.fromFile(filePath)
    try
      for line <- source.getLines() do
        val trimmed = line.trim
        if trimmed.nonEmpty then
          val idx = trimmed.indexOf(':')
          if idx != -1 then
            val name = trimmed.substring(0, idx).trim
            val pType = trimmed.substring(idx + 1).trim
            map(name) = pType
    finally
      source.close()

  def isPlaceName(name: String): Boolean =
    placeNameMap.contains(name)

  def isPlaceName(words: List[String], startIndex: Int, numWords: Int): Boolean =
    if startIndex + numWords > words.size then false
    else
      val s = words.slice(startIndex, startIndex + numWords).mkString(" ")
      isPlaceName(s)

  def isHumanName(words: List[String]): Boolean =
    val len = words.size
    if len == 1 then
      lastNameSet.contains(words(0))
    else if len == 2 then
      (firstNameSet.contains(words(0)) && lastNameSet.contains(words(1))) ||
      (prefixSet.contains(words(0)) && lastNameSet.contains(words(1)))
    else if len == 3 then
      (firstNameSet.contains(words(0)) && firstNameSet.contains(words(1)) && lastNameSet.contains(words(2))) ||
      (prefixSet.contains(words(0)) && firstNameSet.contains(words(1)) && lastNameSet.contains(words(2))) ||
      (prefixSet.contains(words(0)) && words(1) == "." && lastNameSet.contains(words(2)))
    else if len == 4 then
      (firstNameSet.contains(words(0)) && firstNameSet.contains(words(1)) && firstNameSet.contains(words(2)) && lastNameSet.contains(words(3))) ||
      (firstNameSet.contains(words(0)) && words(1).length == 1 && words(2) == "." && lastNameSet.contains(words(3))) ||
      (prefixSet.contains(words(0)) && firstNameSet.contains(words(1)) && firstNameSet.contains(words(2)) && lastNameSet.contains(words(3))) ||
      (prefixSet.contains(words(0)) && firstNameSet.contains(words(1)) && words(2).length == 1 && lastNameSet.contains(words(3)))
    else if len == 5 then
      (firstNameSet.contains(words(0)) && firstNameSet.contains(words(1)) && words(2).length == 1 && words(3) == "." && lastNameSet.contains(words(4))) ||
      (prefixSet.contains(words(0)) && firstNameSet.contains(words(1)) && words(2).length == 1 && words(3) == "." && lastNameSet.contains(words(4)))
    else
      false

  def isHumanName(s: String): Boolean =
    isHumanName(Tokenizer.wordsToList(s))

  def isHumanName(words: List[String], index: Int, numWords: Int): Boolean =
    if index + numWords > words.size then false
    else isHumanName(words.slice(index, index + numWords))


  def getProperNames(words: List[String]): (ScoredList, ScoredList) =
    val humanNames = ScoredList()
    val placeNames = ScoredList()
    var i = 0
    while i < words.size do
      // Check 5-word human names
      if isHumanName(words, i, 5) then
        humanNames.addValue(words.slice(i, i + 5).mkString(" "))
        i += 5
      // Check 4-word human names
      else if isHumanName(words, i, 4) then
        humanNames.addValue(words.slice(i, i + 4).mkString(" "))
        i += 4
      // Check 3-word place names
      else if isPlaceName(words, i, 3) then
        placeNames.addValue(words.slice(i, i + 3).mkString(" "))
        i += 3
      // Check 3-word human names
      else if isHumanName(words, i, 3) then
        humanNames.addValue(words.slice(i, i + 3).mkString(" "))
        i += 3
      // Check 2-word place names
      else if isPlaceName(words, i, 2) then
        placeNames.addValue(words.slice(i, i + 2).mkString(" "))
        i += 2

      // Check 2-word human names
      else if isHumanName(words, i, 2) then
        humanNames.addValue(words.slice(i, i + 2).mkString(" "))
        i += 2
      // Check 1-word place names
      else if isPlaceName(words, i, 1) then
        placeNames.addValue(words(i))
        i += 1
      else
        i += 1
    (humanNames, placeNames)

  def getProperNames(s: String): (ScoredList, ScoredList) =
    getProperNames(Tokenizer.wordsToList(s))
