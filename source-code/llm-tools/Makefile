.PHONY: clean check test run

MODEL ?= mistral

check:
	scala-cli compile .

test:
	scala-cli run . --main-class llmtools.toolsTest

run:
	scala-cli run . --main-class llmtools.toolsDemo -- $(MODEL)

clean:
	scala-cli clean . 2>/dev/null || true
	find . -name ".scala-build" -type d -exec rm -rf {} +
	find . -name ".bsp" -type d -exec rm -rf {} +
