# Makefile for go-playing-program
# Copyright 2026 Mark Watson. All rights reserved.
#
# Two goals shape this file:
#
#   1. work on one part of the system at a time
#   2. run anything with one short command
#
# So every subsystem has its own test and demo target, and the whole thing has
# the usual `check` / `run` / `clean` entry points that the parent Makefile in
# source-code/ expects when it recurses into this directory.
#
# Quick reference
#   make check          compile, then run every test suite
#   make test           run every test suite
#   make run            play against the engine in the terminal
#   make gtp            speak GTP on stdin/stdout
#   make selfplay       generate reinforcement-learning training data
#   make demo-all       a tour of every subsystem
#   make help           the full target list
#
# ---------------------------------------------------------------------------
# Scala CLI setup
# ---------------------------------------------------------------------------
# Bloop is the incremental compiler server.  When its cache is writable it
# makes rebuilds much faster, but in restricted environments (containers, CI,
# some sandboxes) it cannot create its socket or write its cache and every
# invocation fails with a confusing Java exception.  So the server is enabled
# only when the cache directory is actually writable; override with SERVER=0 or
# SERVER=1 to force the decision.
CACHE_DIR := $(HOME)/Library/Caches/ScalaCli

ifeq ($(SERVER),)
  ifeq ($(shell test -w "$(CACHE_DIR)" 2>/dev/null && echo yes),yes)
    SERVER_FLAG :=
  else
    SERVER_FLAG := --server=false
  endif
else ifeq ($(SERVER),0)
  SERVER_FLAG := --server=false
else
  SERVER_FLAG :=
endif

SCALA_CLI ?= scala-cli
RUN := $(SCALA_CLI) run . $(SERVER_FLAG)
COMPILE := $(SCALA_CLI) compile . $(SERVER_FLAG)
TEST := $(SCALA_CLI) test . $(SERVER_FLAG)

# Search strength defaults.  Keep these modest so the demos and tests are
# quick; raise them for real play.
SIZE ?= 9
PLAYOUTS ?= $(if $(filter 9,$(SIZE)),2000,4000)
THREADS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
# A per-move time budget, as an alternative to a fixed playout count:
#   make run SIZE=19 TIME=5s
TIME ?=
TIME_FLAG := $(if $(TIME),--time $(TIME),)
# Draw the board on one line per row instead of two, for 19x19 in a short
# terminal:  make run SIZE=19 COMPACT=1
COMPACT ?=
COMPACT_FLAG := $(if $(COMPACT),--compact,)

# ONNX Runtime, for playing with a trained neural network.  The jar is not a
# dependency of the default build: the engine talks to it reflectively, so
# everything works without it and the network is opt-in.
ONNX_VERSION ?= 1.20.0
ONNX_DEP := com.microsoft.onnxruntime:onnxruntime:$(ONNX_VERSION)
MODEL ?= model.onnx
ONNX_RUN := $(SCALA_CLI) run . $(SERVER_FLAG) --dep $(ONNX_DEP)

# The test entry points, in package go.  These names are the @main method
# names, so they must match exactly: `make selfPlayTest` runs go.selfPlayTest.
SUITES := boardTest evalTest searchTest tacticsTest sgfTest selfPlayTest gtpTest cliTest

DEMOS := board eval sgf search gtp selfplay
DEMO_TARGETS := $(addprefix demo-,$(DEMOS))

.PHONY: all check test run clean fmt fmt-check watch help \
        $(DEMO_TARGETS) demo-all \
        test-all test-board test-eval test-search test-tactics test-sgf test-selfplay test-gtp test-cli \
        test-python \
        gtp selfplay onnx-check onnx-run onnx-gtp onnx-selfplay \
        version clean-cache

all: check

# ---------------------------------------------------------------------------
# The three targets the parent Makefile recurses into
# ---------------------------------------------------------------------------
# `check` compiles and then tests, so it is the one command to run before
# committing.  `run` opens the interactive board.  `clean` removes build output.
check: compile test-all

compile:
	@echo "==> compiling"
	@$(COMPILE)

test: test-all

clean:
	-$(SCALA_CLI) clean . 2>/dev/null || true
	rm -rf .scala-build .bsp
	rm -rf demo-game.sgf demo-selfplay selfplay-* target test-output
	find . -name "*~" -delete

# project.scala already sets the default main class, but naming it here keeps
# this target explicit and matches DESIGN.md section 4.3.
run:
	$(RUN) --main-class go.CLI -- --size $(SIZE) --playouts $(PLAYOUTS) --threads $(THREADS) $(TIME_FLAG) $(COMPACT_FLAG)

# ---------------------------------------------------------------------------
# Tests: one target per subsystem, so a change to one file can be checked alone
# ---------------------------------------------------------------------------
# `compile` is a prerequisite of test-all rather than of each suite, so it runs
# exactly once per make invocation.
test-all: compile $(SUITES)
	@echo ""
	@echo "==> all test suites passed"

$(SUITES):
	@echo "==> $@"
	@$(RUN) --main-class go.$@

# Friendly per-subsystem aliases.
test-board: boardTest
test-eval: evalTest
test-search: searchTest
test-tactics: tacticsTest
test-sgf: sgfTest
test-selfplay: selfPlayTest
test-gtp: gtpTest
test-cli: cliTest

# The training-data reader test.  Stdlib only, so it runs without NumPy.
test-python:
	cd python && python3 -m unittest

.PHONY: $(SUITES)

# ---------------------------------------------------------------------------
# Demos: each subsystem is runnable on its own
# ---------------------------------------------------------------------------
demo-all:
	@$(RUN) --main-class go.demoAll

demo-board:
	@$(RUN) --main-class go.demoBoard

demo-search:
	@$(RUN) --main-class go.demoSearch

demo-eval:
	@$(RUN) --main-class go.demoEval

demo-sgf:
	@$(RUN) --main-class go.demoSgf

demo-gtp:
	@$(RUN) --main-class go.demoGtp

demo-selfplay:
	@$(RUN) --main-class go.demoSelfPlay

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
# GTP on stdin/stdout.  Point a GUI at this, for example:
#   Sabaki  -> Engines -> Add Engine -> $(make -n gtp)
#   gogui-twogtp -black "$(make -n gtp)" -white "$(make -n gtp)" -games 10
gtp:
	$(RUN) --main-class go.GTP -- --size $(SIZE) --playouts $(PLAYOUTS) --threads $(THREADS) $(TIME_FLAG)

gtp-log:
	$(RUN) --main-class go.GTP -- --size $(SIZE) --playouts $(PLAYOUTS) $(TIME_FLAG) --log

# Self-play for reinforcement learning.  Writes training.bin plus one SGF per
# game; see python/load_selfplay.py for reading the data.
GAMES ?= 10
OUTPUT ?= selfplay-$(SIZE)x$(SIZE)

selfplay:
	$(RUN) --main-class go.SelfPlay -- \
		--games $(GAMES) --size $(SIZE) --playouts $(PLAYOUTS) \
		--threads $(THREADS) --output $(OUTPUT)

selfplay-json:
	$(RUN) --main-class go.SelfPlay -- \
		--games $(GAMES) --size $(SIZE) --playouts $(PLAYOUTS) \
		--threads $(THREADS) --output $(OUTPUT) --format json

# ---------------------------------------------------------------------------
# Playing with a trained neural network (opt-in, needs the ONNX Runtime jar)
# ---------------------------------------------------------------------------
# The engine loads ONNX models reflectively, so nothing is downloaded unless
# you ask for it.  On Apple Silicon `addCoreML()` selects the Metal backend.
# Verify the setup before playing: reports whether ONNX Runtime is on the
# classpath and, given a model, evaluates the opening position once.
onnx-check:
	@$(ONNX_RUN) --main-class go.onnxCheck -- --model $(MODEL) --size $(SIZE)

onnx-run:
	$(ONNX_RUN) -- --model $(MODEL) --size $(SIZE) --playouts $(PLAYOUTS)

onnx-gtp:
	$(ONNX_RUN) --main-class go.GTP -- --model $(MODEL) --size $(SIZE) --playouts $(PLAYOUTS)

onnx-selfplay:
	$(ONNX_RUN) --main-class go.SelfPlay -- --model $(MODEL) \
		--games $(GAMES) --size $(SIZE) --playouts $(PLAYOUTS) --output $(OUTPUT)

# ---------------------------------------------------------------------------
# Development helpers
# ---------------------------------------------------------------------------
# `make watch` recompiles on every change.  Bloop makes this genuinely
# incremental, so it needs a writable server; without one it polls instead.
watch:
	$(SCALA_CLI) compile . --watch

# scalafmt is optional: if it is not configured the target explains itself
# rather than failing.
fmt:
	@if [ -f .scalafmt.conf ]; then \
		$(SCALA_CLI) fmt . $(SERVER_FLAG); \
	else \
		echo "no .scalafmt.conf; create one to enable formatting"; \
	fi

fmt-check:
	@if [ -f .scalafmt.conf ]; then \
		$(SCALA_CLI) fmt --check . $(SERVER_FLAG); \
	else \
		echo "no .scalafmt.conf; create one to enable formatting checks"; \
	fi

version:
	@$(SCALA_CLI) version
	@java -version 2>&1 | head -1

clean-cache:
	rm -rf $(CACHE_DIR)

help:
	@echo "go-playing-program -- a Scala 3 Go engine with MCTS and neural evaluation"
	@echo ""
	@echo "  make check            compile and run every test suite"
	@echo "  make test             run every test suite"
	@echo "  make run              play against the engine in the terminal"
	@echo "  make gtp              speak GTP on stdin/stdout"
	@echo "  make selfplay         generate training data for the neural network"
	@echo ""
	@echo "One subsystem at a time:"
	@echo "  make test-board       board rules, captures, ko, scoring   (Board.scala)"
	@echo "  make test-eval        feature planes, heuristic policy    (Evaluate.scala)"
	@echo "  make test-search      Monte Carlo tree search            (Search.scala)"
	@echo "  make test-tactics     forced-capture reading             (Tactics.scala)"
	@echo "  make test-sgf         SGF reading and writing             (SGF.scala)"
	@echo "  make test-selfplay    reinforcement-learning data    (SelfPlay.scala)"
	@echo "  make test-gtp         Go Text Protocol                   (GTP.scala)"
	@echo "  make test-cli         board layout and alignment          (CLI.scala)"
	@echo "  make test-python      training-data reader           (python/)"
	@echo ""
	@echo "  make demo-board       rules, shown move by move"
	@echo "  make demo-search      MCTS visits, timing, principal variation"
	@echo "  make demo-eval        feature planes, policy, influence map"
	@echo "  make demo-sgf         write, read and replay game records"
	@echo "  make demo-gtp         a scripted GTP session"
	@echo "  make demo-selfplay    one game's worth of training data"
	@echo "  make demo-all         all of the above"
	@echo ""
	@echo "Options (all overridable on the command line):"
	@echo "  SIZE=9|13|19          board size                     (now $(SIZE))"
	@echo "  PLAYOUTS=N            MCTS playouts per move         (now $(PLAYOUTS))"
	@echo "  TIME=5s               time budget per move           (now '$(TIME)')"
	@echo "  COMPACT=1             one line per board row          (now '$(COMPACT)')"
	@echo "  THREADS=N             search threads                 (now $(THREADS))"
	@echo "  GAMES=N               self-play games                (now $(GAMES))"
	@echo "  OUTPUT=DIR            self-play output directory     (now $(OUTPUT))"
	@echo "  MODEL=FILE            ONNX model for the onnx-* targets"
	@echo "  ONNX_VERSION=V        onnxruntime version            (now $(ONNX_VERSION))"
	@echo "  SERVER=0|1            force the Bloop server off/on  (now '$(SERVER_FLAG)')"
	@echo ""
	@echo "  make watch            recompile on every source change"
	@echo "  make clean            remove build output"
