package nsk

import scala.collection.mutable.ListBuffer

/** A parsed top-level form. */
enum Form:
  case Cmd(name: String, args: List[Term])          // (:add s p o) or a bare :count
  case Pat(triple: Triple)                          // [s p o]
  case Ask(vars: List[Term.Var], clauses: List[Triple]) // (ask (?a) [..] [..])

/** Thrown when the input runs out in the middle of a form. The REPL treats
  * this as "read another line", the way a Lisp reader waits for more input. */
final class IncompleteInput extends RuntimeException("incomplete input")

/** Thrown on malformed input. */
final class ReadError(msg: String) extends RuntimeException(msg)

/** The NSK reader.
  *
  * This is the Scala answer to the Common Lisp reader macros. It turns the
  * same surface syntax into an abstract syntax tree:
  *
  *   ?name    -> Term.Var(name)
  *   :name    -> Term.Kw(NAME)          (upper-cased, as in Common Lisp)
  *   ~term    -> Term.Neural(term)
  *   [s p o]  -> Form.Pat(Triple(...))
  *   (:op ..) -> Form.Cmd(op, args)
  *   (ask ..) -> Form.Ask(vars, clauses)
  */
object Reader:
  /** Read exactly one form from S (trailing text is ignored). */
  def readForm(s: String): Form =
    val c = Cursor(s)
    c.form()

  /** Read as many complete forms as possible; return them and the index of
    * the first unconsumed character (an incomplete trailing form is left). */
  def readForms(s: String): (List[Form], Int) =
    val c    = Cursor(s)
    val out  = ListBuffer.empty[Form]
    var more = true
    while more do
      c.skipWs()
      if c.atEnd then more = false
      else
        val mark = c.pos
        try out += c.form()
        catch
          case _: IncompleteInput =>
            c.pos = mark
            more = false
    (out.toList, c.pos)

  private final class Cursor(s: String):
    var pos = 0

    def atEnd: Boolean = pos >= s.length
    def peek: Char     = s(pos)

    def skipWs(): Unit = while pos < s.length && s(pos).isWhitespace do pos += 1

    private def expect(ch: Char): Unit =
      skipWs()
      if atEnd then throw IncompleteInput()
      if s(pos) == ch then pos += 1 else throw ReadError(s"expected '$ch' at $pos")

    private def tokenChars(): String =
      val start = pos
      while pos < s.length && !s(pos).isWhitespace && !"()[]".contains(s(pos)) do pos += 1
      if pos == start then throw ReadError(s"empty token at $pos")
      s.substring(start, pos)

    /** A single term. */
    def term(): Term =
      skipWs()
      if atEnd then throw IncompleteInput()
      peek match
        case '?'                        => pos += 1; Term.Var(tokenChars())
        case ':'                        => pos += 1; Term.Kw(tokenChars().toUpperCase)
        case '~'                        => pos += 1; Term.Neural(term())
        case '"'                        => Term.Str(stringLiteral())
        case c if c == '-' || c.isDigit => number()
        case c                          => throw ReadError(s"unexpected term char '$c' at $pos")

    private def number(): Term =
      val start = pos
      if peek == '-' then pos += 1
      while pos < s.length && s(pos).isDigit do pos += 1
      Term.Num(s.substring(start, pos).toLong)

    private def stringLiteral(): String =
      expect('"')
      val sb   = StringBuilder()
      var done = false
      while !done do
        if atEnd then throw IncompleteInput()
        val ch = s(pos); pos += 1
        ch match
          case '"'  => done = true
          case '\\' =>
            if atEnd then throw IncompleteInput()
            val e = s(pos); pos += 1
            sb.append(e match
              case 'n'   => '\n'
              case 't'   => '\t'
              case 'r'   => '\r'
              case other => other
            )
          case c => sb.append(c)
      sb.toString

    /** A [s p o] triple pattern. */
    def triple(): Triple =
      expect('[')
      val terms = termsUntil(']')
      if terms.length != 3 then
        throw ReadError(s"triple pattern needs exactly three terms, got ${terms.length}")
      Triple(terms(0), terms(1), terms(2))

    private def termsUntil(close: Char): List[Term] =
      val out  = ListBuffer.empty[Term]
      var done = false
      while !done do
        skipWs()
        if atEnd then throw IncompleteInput()
        if peek == close then { pos += 1; done = true }
        else out += term()
      out.toList

    private def clausesUntil(close: Char): List[Triple] =
      val out  = ListBuffer.empty[Triple]
      var done = false
      while !done do
        skipWs()
        if atEnd then throw IncompleteInput()
        peek match
          case c if c == close => pos += 1; done = true
          case '['             => out += triple()
          case _               => throw ReadError(s"ask clause is not a [triple] at $pos")
      out.toList

    private def bareWord(): String =
      skipWs()
      val start = pos
      while pos < s.length && !s(pos).isWhitespace && !"()[]".contains(s(pos)) do pos += 1
      s.substring(start, pos)

    /** A top-level form. */
    def form(): Form =
      skipWs()
      if atEnd then throw IncompleteInput()
      peek match
        case '(' => listForm()
        case '[' => Form.Pat(triple())
        case ':' => pos += 1; Form.Cmd(tokenChars().toLowerCase, Nil)
        case c   => throw ReadError(s"unexpected char '$c' at $pos")

    private def listForm(): Form =
      expect('(')
      skipWs()
      if atEnd then throw IncompleteInput()
      if peek == ':' then
        pos += 1
        val name = tokenChars().toLowerCase
        Form.Cmd(name, termsUntil(')'))
      else
        bareWord() match
          case "ask" =>
            skipWs(); expect('(')
            val vars    = termsUntil(')').collect { case v: Term.Var => v }
            val clauses = clausesUntil(')')
            Form.Ask(vars, clauses)
          case head => throw ReadError(s"unknown list head '$head'")
