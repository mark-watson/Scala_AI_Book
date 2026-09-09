// Copyright 2026 Mark Watson. All rights reserved.

package webapp

import com.raquo.laminar.api.L.{*, given}
import org.scalajs.dom

// App entry point. scala-cli links this @main into main.js, which runs
// it on page load. Mounts the two demo widgets under #app.
@main def webApp(): Unit =
  val container = dom.document.querySelector("#app")
  require(container != null, "Missing <div id=\"app\"> in index.html.")
  render(
    container,
    div(
      cls := "page",
      h1("Scala AI Book: interactive widgets"),
      p("Two small Laminar widgets. Bayes meets the probability chapter, search meets the RAG chapter."),
      BayesCalc.widget,
      DocFilter.widget
    )
  )
