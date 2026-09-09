.PHONY: clean check test run

check:
	scala-cli compile .

test:
	scala-cli run . --main-class symbolicmath.symbolicMathTest

run:
	scala-cli run . --main-class symbolicmath.symbolicMathDemo

clean:
	scala-cli clean . 2>/dev/null || true
	find . -name ".scala-build" -type d -exec rm -rf {} +
	find . -name ".bsp" -type d -exec rm -rf {} +
