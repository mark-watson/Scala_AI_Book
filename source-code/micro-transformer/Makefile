.PHONY: clean check test run

check:
	scala-cli compile .

test:
	scala-cli run . --main-class microtransformer.transformerTest

run:
	scala-cli run . --main-class microtransformer.microDemo

clean:
	scala-cli clean . 2>/dev/null || true
	find . -name ".scala-build" -type d -exec rm -rf {} +
	find . -name ".bsp" -type d -exec rm -rf {} +
