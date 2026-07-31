.SHELL := /bin/bash
SCALA_CLI := scala-cli
SCALA_FLAGS := --server=false
MODEL ?= mistral

.PHONY: all compile run test fmt clean test-ollama

all: compile

compile:
	$(SCALA_CLI) compile . $(SCALA_FLAGS)

run:
	$(SCALA_CLI) run . $(SCALA_FLAGS) --main-class textadventure.play

test:
	$(SCALA_CLI) test . $(SCALA_FLAGS)

fmt:
	$(SCALA_CLI) fmt . $(SCALA_FLAGS)

test-ollama:
	$(SCALA_CLI) run . $(SCALA_FLAGS) --main-class textadventure.testOllama -- $(MODEL)

clean:
	rm -rf .scala-build .bsp .poolside
