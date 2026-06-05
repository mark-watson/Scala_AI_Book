// Copyright 2025-2026 Mark Watson. All rights reserved.

package nlp

import scala.io.Source
import java.io.File

class FastTag(dataPath: String = "data/lexicon.txt"):
  private val lexicon = collection.mutable.Map[String, Array[String]]()

  // Load lexicon
  private val resolvedPath = 
    if File(dataPath).exists() then dataPath
    else if File("nlp/" + dataPath).exists() then "nlp/" + dataPath
    else throw java.io.FileNotFoundException(s"Lexicon file not found at $dataPath or nlp/$dataPath")

  val source = Source.fromFile(resolvedPath)
  try
    for line <- source.getLines() do
      val parts = line.split(" ")
      if parts.length > 1 then
        lexicon(parts(0)) = parts.tail
  finally
    source.close()

  def wordInLexicon(word: String): Boolean =
    lexicon.contains(word) || lexicon.contains(word.toLowerCase)

  def tag(words: List[String]): List[String] =
    val ret = collection.mutable.ListBuffer[String]()
    for word <- words do
      val ss = lexicon.get(word).orElse(lexicon.get(word.toLowerCase))
      ss match
        case Some(tags) => ret.append(tags(0))
        case None =>
          if word.length == 1 then ret.append(word + "^")
          else ret.append("NN")

    val result = ret.toArray
    // Apply transformational rules
    for i <- words.indices do
      val word = result(i)
      // rule 1: DT, {VBD | VBP | VB} --> DT, NN
      if i > 0 && result(i - 1) == "DT" then
        if word == "VBD" || word == "VBP" || word == "VB" then
          result(i) = "NN"

      // rule 2: convert a noun to a number (CD) if "." appears in the word or it parses as a float
      if result(i).startsWith("N") then
        if words(i).contains(".") then
          result(i) = "CD"
        else
          try
            words(i).toFloat
            result(i) = "CD"
          catch
            case _: NumberFormatException => // ignore

      // rule 3: convert a noun to a past participle if word ends with "ed"
      if result(i).startsWith("N") && words(i).endsWith("ed") then
        result(i) = "VBN"

      // rule 4: convert any type to adverb if it ends in "ly"
      if words(i).endsWith("ly") then
        result(i) = "RB"

      // rule 5: convert a common noun (NN or NNS) to an adjective if it ends with "al"
      if result(i).startsWith("NN") && words(i).endsWith("al") then
        result(i) = "JJ"

      // rule 6: convert a noun to a verb if the preceding word is "would"
      if i > 0 && result(i).startsWith("NN") && words(i - 1).equalsIgnoreCase("would") then
        result(i) = "VB"

      // rule 7: convert NN to NNS if it ends with "s"
      if result(i) == "NN" && words(i).endsWith("s") then
        result(i) = "NNS"

      // rule 8: convert common noun to present participle verb (gerund) if it ends with "ing"
      if result(i).startsWith("NN") && words(i).endsWith("ing") then
        result(i) = "VBG"

    result.toList
