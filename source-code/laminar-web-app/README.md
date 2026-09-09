# Laminar Web App — Scala AI Book

Two interactive widgets built with Laminar on Scala.js, compiled with
plain Scala CLI: a live Bayes posterior calculator and a reactive doc
filter. No sbt project, no Node, no npm, no bundler.

## Run

Needs network once to fetch deps, then a browser:

    scala-cli --power package . --js -o main.js -f
    python3 -m http.server 8000

Open http://localhost:8000/index.html in a browser.

## Check

    scala-cli compile . --js

## Book Cover Material, Copyright, and License

This example code is released using the Apache 2 license.

Copyright 2026 Mark Watson. All rights reserved.
