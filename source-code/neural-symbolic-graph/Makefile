# NSK: Neural-Symbolic Knowledge Graph engine (Scala)
#
# Wraps scala-cli. JVM defaults to `system`, so scala-cli uses the JDK already
# on your PATH instead of downloading one. Override with `make JVM=temurin:21 …`.

SCALA_CLI ?= scala-cli
JVM       ?= system
FLAGS     := --jvm $(JVM)

.PHONY: help compile test run repl serve fmt clean

help:
	@echo "Targets:"
	@echo "  compile  type-check all sources"
	@echo "  test     run the test suite"
	@echo "  run      start the interactive REPL"
	@echo "  serve    start the REST server on port 8800"
	@echo "  fmt      format sources with scalafmt"
	@echo "  clean    remove build artifacts and log files"

compile:
	$(SCALA_CLI) compile . $(FLAGS)

test:
	$(SCALA_CLI) run . $(FLAGS) --main-class nsk.test

repl:
	$(SCALA_CLI) run . $(FLAGS)

serve:
	$(SCALA_CLI) run . $(FLAGS) -- --serve --port 8800

fmt:
	$(SCALA_CLI) fmt .

clean:
	$(SCALA_CLI) clean .
	rm -f nsk-graph.log nsk-test-tmp.log
