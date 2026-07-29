//> using scala 3.8.4
//> using mainClass nsk.main

// NSK: Neural-Symbolic Knowledge Graph engine, ported from Common Lisp.
//
// The core has no external dependencies. JSON is hand-rolled, the Ollama
// client uses the JDK HttpClient, and the REST server uses the JDK's
// built-in com.sun.net.httpserver. Build and run with scala-cli:
//
//   scala-cli run .                         start the REPL
//   scala-cli run . -- --serve --port 8800  start the REST server
//   scala-cli run . --main-class nsk.test   run the test suite
