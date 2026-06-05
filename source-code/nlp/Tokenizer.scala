// Copyright 2025-2026 Mark Watson. All rights reserved.
//> using scala 3.6.4

package nlp

import java.io.{StringReader, StreamTokenizer, IOException}
import scala.collection.mutable.ListBuffer

object Tokenizer:

  def wordsToList(s2: String): List[String] =
    wordsToList(s2, s2.length + 1)

  def wordsToList(s2: String, maxR: Int): List[String] =
    val clean = stripControlCharacters(s2)
    val words = ListBuffer[String]()
    var count = 0
    try
      val strTok = StreamTokenizer(StringReader(clean))
      strTok.whitespaceChars('"', '"')
      strTok.whitespaceChars('\'', '\'')
      strTok.whitespaceChars('/', '/')

      while strTok.nextToken() != StreamTokenizer.TT_EOF && count < maxR do
        val s = strTok.ttype match
          case StreamTokenizer.TT_EOL => ""
          case StreamTokenizer.TT_WORD => strTok.sval
          case StreamTokenizer.TT_NUMBER => strTok.nval.toInt.toString
          case ttype => ttype.toChar.toString

        if s.nonEmpty then
          if s.endsWith(".") then
            val index = s.indexOf(".")
            if index < s.length - 1 then
              words.append(s)
            else
              words.append(s.substring(0, s.length - 1))
              words.append(".")
          else if s.endsWith(",") then
            val x = s.substring(0, s.length - 1)
            if x.nonEmpty then words.append(x)
            words.append(",")
          else if s.endsWith(";") then
            val x = s.substring(0, s.length - 1)
            if x.nonEmpty then words.append(x)
            words.append(";")
          else if s.endsWith("?") then
            val x = s.substring(0, s.length - 1)
            if x.nonEmpty then words.append(x)
            words.append("?")
          else if s.endsWith(":") then
            val x = s.substring(0, s.length - 1)
            if x.nonEmpty then words.append(x)
            words.append(":")
          else
            words.append(s)
          count += 1
    catch
      case e: IOException => e.printStackTrace()

    words.toList

  private def stripControlCharacters(s: String): String =
    val sb = StringBuilder(s.length + 1)
    for i <- 0 until s.length do
      val ch = s.charAt(i)
      if ch > 256 || ch == '\n' || ch == '\t' || ch == '\r' || ch == 226 then
        sb.append(' ')
      else if ch.toInt < 129 then
        sb.append(ch)
      else
        sb.append(' ')
    sb.toString
