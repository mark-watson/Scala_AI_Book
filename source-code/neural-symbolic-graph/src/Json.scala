package nsk

/** A small, self-contained JSON model, parser, and writer.
  *
  * NSK keeps its own JSON code so the core has no external dependency, the
  * same choice the Common Lisp version makes. The Lisp writer takes a tagged
  * list and the reader returns alists; here JSON is a proper algebraic data
  * type, which is the idiomatic Scala model.
  */
enum Json:
  case Str(value: String)
  case Num(value: Double)
  case Bool(value: Boolean)
  case Null
  case Arr(items: List[Json])
  case Obj(fields: List[(String, Json)])

  def render: String =
    val sb = StringBuilder()
    Json.write(this, sb)
    sb.toString

  /** Look up KEY in an object, the analog of Common Lisp `json-get`. */
  def apply(key: String): Option[Json] = this match
    case Obj(fs) => fs.collectFirst { case (k, v) if k == key => v }
    case _       => None

  def str: Option[String]    = this match { case Str(s) => Some(s); case _ => None }
  def double: Option[Double] = this match { case Num(n) => Some(n); case _ => None }

object Json:
  def obj(fields: (String, Json)*): Json = Obj(fields.toList)
  def arr(items: Json*): Json            = Arr(items.toList)

  final class JsonError(msg: String) extends RuntimeException(msg)

  def parse(input: String): Json =
    val p = Parser(input)
    val v = p.value()
    p.skipWs()
    v

  // ----- writer -----------------------------------------------------------

  private def write(v: Json, sb: StringBuilder): Unit = v match
    case Str(s)  => writeString(s, sb)
    case Num(n)  => if n == n.toLong.toDouble then sb.append(n.toLong) else sb.append(n)
    case Bool(b) => sb.append(if b then "true" else "false")
    case Null    => sb.append("null")
    case Arr(items) =>
      sb.append('[')
      items.zipWithIndex.foreach { (it, i) => if i > 0 then sb.append(','); write(it, sb) }
      sb.append(']')
    case Obj(fields) =>
      sb.append('{')
      fields.zipWithIndex.foreach { case ((k, vv), i) =>
        if i > 0 then sb.append(',')
        writeString(k, sb); sb.append(':'); write(vv, sb)
      }
      sb.append('}')

  private def writeString(s: String, sb: StringBuilder): Unit =
    sb.append('"')
    s.foreach {
      case '"'          => sb.append("\\\"")
      case '\\'         => sb.append("\\\\")
      case '\n'         => sb.append("\\n")
      case '\r'         => sb.append("\\r")
      case '\t'         => sb.append("\\t")
      case '\b'         => sb.append("\\b")
      case '\f'         => sb.append("\\f")
      case c if c < ' ' => sb.append("\\u%04x".format(c.toInt))
      case c            => sb.append(c)
    }
    sb.append('"')

  // ----- reader -----------------------------------------------------------

  private final class Parser(s: String):
    private var pos = 0

    private def fail(msg: String): Nothing = throw JsonError(s"$msg at $pos")
    private def atEnd: Boolean             = pos >= s.length
    private def peek: Char                 = s(pos)
    private def next(): Char               = { val c = s(pos); pos += 1; c }

    def skipWs(): Unit =
      while pos < s.length && " \t\n\r".contains(s(pos)) do pos += 1

    def value(): Json =
      skipWs()
      if atEnd then fail("unexpected end of input")
      peek match
        case '{'                       => obj()
        case '['                       => arr()
        case '"'                       => Str(string())
        case 't'                       => literal("true", Bool(true))
        case 'f'                       => literal("false", Bool(false))
        case 'n'                       => literal("null", Null)
        case c if c == '-' || c.isDigit => number()
        case c                         => fail(s"unexpected char $c")

    private def literal(text: String, v: Json): Json =
      text.foreach(e => if atEnd || next() != e then fail(s"bad literal, expected $text"))
      v

    private def obj(): Json =
      next() // {
      skipWs()
      if !atEnd && peek == '}' then { next(); return Obj(Nil) }
      val fields = List.newBuilder[(String, Json)]
      var more   = true
      while more do
        skipWs()
        val key = string()
        skipWs()
        if atEnd || next() != ':' then fail("expected : after object key")
        fields += (key -> value())
        skipWs()
        next() match
          case ',' => ()
          case '}' => more = false
          case _   => fail("expected , or } in object")
      Obj(fields.result())

    private def arr(): Json =
      next() // [
      skipWs()
      if !atEnd && peek == ']' then { next(); return Arr(Nil) }
      val items = List.newBuilder[Json]
      var more  = true
      while more do
        items += value()
        skipWs()
        next() match
          case ',' => ()
          case ']' => more = false
          case _   => fail("expected , or ] in array")
      Arr(items.result())

    private def string(): String =
      if atEnd || next() != '"' then fail("expected a string")
      val sb   = StringBuilder()
      var done = false
      while !done do
        if atEnd then fail("unterminated string")
        next() match
          case '"'  => done = true
          case '\\' =>
            next() match
              case '"'  => sb.append('"')
              case '\\' => sb.append('\\')
              case '/'  => sb.append('/')
              case 'b'  => sb.append('\b')
              case 'f'  => sb.append('\f')
              case 'n'  => sb.append('\n')
              case 'r'  => sb.append('\r')
              case 't'  => sb.append('\t')
              case 'u'  => sb.append(hex(4).toChar)
              case _    => fail("bad string escape")
          case c => sb.append(c)
      sb.toString

    private def hex(n: Int): Int =
      var v = 0
      for _ <- 0 until n do
        val d = Character.digit(next(), 16)
        if d < 0 then fail("bad \\u escape")
        v = v * 16 + d
      v

    private def number(): Json =
      val start = pos
      if peek == '-' then next()
      while !atEnd && (peek.isDigit || "+-.eE".contains(peek)) do next()
      Num(s.substring(start, pos).toDouble)
