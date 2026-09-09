// Copyright 2026 Mark Watson. All rights reserved.

package webapp

import com.raquo.laminar.api.L.{*, given}

// Live Bayes posterior calculator. Same math as source-code/probability,
// but each slider drag pushes a new value through Airstream signals and
// Laminar redraws only the text and bar that changed. No virtual DOM,
// no refresh button, no manual DOM writes.
object BayesCalc:
  private val priorVar = Var(0.001) // P(disease), the base rate
  private val sensVar = Var(0.99)   // P(positive | disease)
  private val fprVar = Var(0.05)    // P(positive | healthy)

  private val posterior: Signal[Double] =
    priorVar.signal.combineWith(sensVar.signal, fprVar.signal).map { case (prev, sens, fpr) =>
      val pPositive = sens * prev + fpr * (1.0 - prev)
      if pPositive == 0.0 then 0.0 else sens * prev / pPositive
    }

  private def slider(labelText: String, v: Var[Double], max: Double, step: Double): Div =
    div(
      cls := "row",
      label(cls := "slider-label", labelText),
      input(
        typ := "range",
        minAttr := "0",
        maxAttr := max.toString,
        stepAttr := step.toString,
        controlled(
          value <-- v.signal.map(_.toString),
          onInput.mapToValue.map(_.toDoubleOption.getOrElse(0.0)) --> v.writer
        )
      ),
      span(cls := "slider-value", child.text <-- v.signal.map(x => f"$x%.4f"))
    )

  def widget: Div =
    div(
      cls := "card",
      h2("Bayes posterior calculator"),
      p("Drag a slider. The chance of disease given a positive test updates at once."),
      slider("Base rate P(disease)", priorVar, 0.05, 0.0005),
      slider("Sensitivity P(positive | disease)", sensVar, 1.0, 0.005),
      slider("False positive rate", fprVar, 0.3, 0.005),
      div(
        cls := "row",
        strong("P(disease | positive) = "),
        span(cls := "result", child.text <-- posterior.map(p => f"${p * 100}%.2f%%"))
      ),
      div(cls := "bar-track", div(cls := "bar-fill", width <-- posterior.map(p => s"${p * 100}%")))
    )
