# Interactive Web Widgets with Laminar

So far each example runs on the shell. At times you want a page: sliders that retune a model, boxes that filter docs, bars that move with the numbers. This chapter builds two such widgets with Laminar on Scala.js, compiled with plain Scala CLI. No sbt project, no Node, no npm, no bundler. Readers write Scala, package one JS file, and open it in a browser.

All code is in `source-code/laminar-web-app`.

## Why Laminar Fits This Book

Laminar skips the virtual DOM. UI state lives in Airstream observables: `Var` holds a value you can set, `Signal` reads values that shift over time, and binders like `<--` and `-->` wire signals to the page. That shape will feel known if you read the search or agent chapters: state flows one way, and each view reads only what it shows. Three sliders drive one posterior number. One text box drives one list. No callbacks to untangle.

Two pinned deps carry the stack: Laminar 17.2.1 for widgets plus Airstream, and scalajs-dom 2.8.1 for `dom.document`. Both live in `project.scala`, the one place directives sit:

```scala
//> using scala 3.6.4
//> using platform scala-js
//> using dep com.raquo::laminar::17.2.1
//> using dep org.scala-js::scalajs-dom::2.8.1
```

## Widget One: Bayes Sliders

`BayesCalc` ports the probability chapter math to sliders. Base rate, sensitivity, and false positive rate each get a `Var[Double]` plus a range input locked to it through a `controlled` block. Locking matters: with it the slider is the sole truth for its value, and user drags plus signal writes never fight. The three signals join with `combineWith` into one posterior, which feeds both the readout text and the bar width:

```scala
controlled(
  value <-- v.signal.map(_.toString),
  onInput.mapToValue.map(_.toDoubleOption.getOrElse(0.0)) --> v.writer
)
```

![Screen shot of web app](laminar_web_app.jpg)

Set all three sliders to the book values (0.001, 0.99, 0.05) and the page shows 1.94%, the same number the shell test pins.

## Widget Two: Doc Search

`DocFilter` holds five doc lines and a query string. Each keystroke maps the query to a match list, and the `split` binder keys rows by text so Laminar keeps, drops, or adds only changed rows:

```scala
ul(children <-- matches.split(identity)((_, doc, _) => li(doc)))
```

The count line reads the same signal, so it can never drift from the rows.

## Build and Open

Package with Scala CLI, serve the dir over http (plain file open starves some browsers of script rights), and load the page:

```bash
cd source-code/laminar-web-app
scala-cli --power package . --js -o main.js -f
python3 -m http.server 8000
```

Open http://localhost:8000/index.html. `Main.scala` mounts both widgets under `#app` on page load. Check the wiring with `scala-cli compile . --js`, which needs no browser. When a widget outgrows this shape, move the same sources to an sbt plus Vite setup; the Laminar code stays put.
