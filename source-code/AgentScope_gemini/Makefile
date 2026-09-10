.PHONY: clean check run

check:
	scala-cli compile .

run:
	scala-cli run . --main-class agentscope_gemini.agentScopeDemo

clean:
	scala-cli clean . 2>/dev/null || true
	find . -name ".scala-build" -type d -exec rm -rf {} +
	find . -name ".bsp" -type d -exec rm -rf {} +
