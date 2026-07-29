package nsk

import java.io.{BufferedReader, File, FileReader, FileWriter, PrintWriter}
import scala.collection.mutable

/** The triplestore: two hash indices plus an append-only log.
  *
  * Triples index by subject and by object for fast pattern narrowing. State
  * survives restarts through a log of forms that is replayed on startup. The
  * log uses the NSK surface syntax, so [[Reader]] replays it and no `eval` is
  * involved, which makes replay safe by construction.
  */
final class Graph private (val logPath: Option[String]):
  private val spo  = mutable.Map.empty[Term, List[Triple]] // subject -> triples
  private val osp  = mutable.Map.empty[Term, List[Triple]] // object  -> triples
  private var live = List.empty[Triple]                    // every live triple, newest first
  private var n    = 0
  private var log  = Option.empty[PrintWriter]

  def count: Int                    = n
  def all: List[Triple]             = live.reverse         // insertion order
  def spoIndex(k: Term): List[Triple] = spo.getOrElse(k, Nil)
  def ospIndex(k: Term): List[Triple] = osp.getOrElse(k, Nil)

  private def index(tr: Triple): Unit =
    spo(tr.s) = tr :: spoIndex(tr.s)
    osp(tr.o) = tr :: ospIndex(tr.o)
    live = tr :: live
    n += 1

  private def unindex(tr: Triple): Unit =
    spo(tr.s) = spoIndex(tr.s).filterNot(_ == tr)
    osp(tr.o) = ospIndex(tr.o).filterNot(_ == tr)
    live = live.filterNot(_ == tr)
    n -= 1

  def present(tr: Triple): Boolean = spoIndex(tr.s).contains(tr)

  private def writeLog(op: String, tr: Triple): Unit =
    log.foreach { w => w.println(Graph.entry(op, tr)); w.flush() }

  /** Add a triple and append it to the log. Duplicates are ignored. */
  def add(s: Term, p: Term, o: Term): Triple =
    val tr = Triple(s, p, o)
    if !present(tr) then { index(tr); writeLog("add", tr) }
    tr

  /** Remove a triple and record the deletion in the log. */
  def remove(s: Term, p: Term, o: Term): Triple =
    val tr = Triple(s, p, o)
    if present(tr) then { unindex(tr); writeLog("del", tr) }
    tr

  /** Triples that could match PATTERN, narrowed by the indices. */
  def candidates(pat: Triple): List[Triple] =
    if Term.concrete(pat.s) then spoIndex(pat.s)
    else if Term.concrete(pat.o) then ospIndex(pat.o)
    else live

  private def applyEntry(form: Form): Unit = form match
    case Form.Cmd("add", List(s, p, o)) =>
      val tr = Triple(s, p, o); if !present(tr) then index(tr)
    case Form.Cmd("del", List(s, p, o)) =>
      val tr = Triple(s, p, o); if present(tr) then unindex(tr)
    case _ => ()

  private def replay(path: String): Unit =
    val f = File(path)
    if f.exists then
      val in = BufferedReader(FileReader(f))
      try
        var line = in.readLine()
        while line != null do
          val t = line.trim
          if t.nonEmpty then applyEntry(Reader.readForm(t))
          line = in.readLine()
      finally in.close()

  private def openLog(path: String): Unit =
    log = Some(PrintWriter(FileWriter(path, /* append = */ true)))

  def flush(): Unit = log.foreach(_.flush())

  def close(): Unit =
    log.foreach { w => w.flush(); w.close() }
    log = None

object Graph:
  /** A fresh in-memory graph with no backing log. */
  def inMemory: Graph = new Graph(None)

  /** Open (or create) the store at PATH, replay its log, keep it open. */
  def open(path: String): Graph =
    val g = new Graph(Some(path))
    g.replay(path)
    g.openLog(path)
    g

  private def entry(op: String, tr: Triple): String =
    s"(:$op ${source(tr.s)} ${source(tr.p)} ${source(tr.o)})"

  private def source(t: Term): String = t match
    case Term.Kw(name) => ":" + name.toLowerCase
    case Term.Str(s)   => "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
    case Term.Num(x)   => x.toString
    case other         => other.toString
