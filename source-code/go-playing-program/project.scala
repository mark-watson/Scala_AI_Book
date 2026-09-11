//> using scala 3.6.4
//> using options -deprecation -feature
// Copyright 2026 Mark Watson. All rights reserved.

// Shared scala-cli build configuration for the whole project.
//
// This file contains ONLY `//> using` directives, which is the scala-cli
// convention for centralising build settings.  Every source file also carries
// its own `//> using scala` line so that any single file stays compilable on
// its own (see README.md, "Working on one part at a time").
//
// Deliberately absent: any `//> using dep` for the neural network.  The core
// engine has ZERO runtime dependencies and the ONNX backend is opt-in; see
// Evaluate.scala and the `make onnx-*` targets in the Makefile.
//
// `mainClass` makes a bare `scala-cli run .` open the interactive CLI.  Every
// other entry point is still reachable with an explicit `--main-class`, which
// overrides this default.

//> using mainClass go.CLI
