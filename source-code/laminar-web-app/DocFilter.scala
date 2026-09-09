// Copyright 2026 Mark Watson. All rights reserved.

package webapp

import com.raquo.laminar.api.L.{*, given}

// Reactive doc filter. Typing in the box narrows the list below through
// a Signal[List[String]]. The split operator keys rows by text, so
// Laminar keeps, drops, or adds only the rows that changed.
object DocFilter:
  private val docs = List(
    "Solar panels turn sunlight into electricity with no exhaust.",
    "A home charger adds about 40 miles of range per hour.",
    "Carbon dioxide from fossil fuel traps heat in the air.",
    "Grid batteries store spare noon solar for the evening peak.",
    "Used battery packs get a second life as home storage."
  )

  private val query = Var("")

  private val matches: Signal[List[String]] =
    query.signal.map { q =>
      val t = q.trim.toLowerCase
      if t.isEmpty then docs else docs.filter(_.toLowerCase.contains(t))
    }

  def widget: Div =
    div(
      cls := "card",
      h2("Search the docs"),
      p("Type to filter. The count and the rows follow each keystroke."),
      input(
        typ := "text",
        placeholder := "Type to filter...",
        controlled(value <-- query.signal, onInput.mapToValue --> query.writer)
      ),
      p(child.text <-- matches.map(ms => s"${ms.size} of ${docs.size} docs match.")),
      ul(children <-- matches.split(identity)((_, doc, _) => li(doc)))
    )
