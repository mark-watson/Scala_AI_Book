.PHONY: clean check run

check:
	scala-cli compile . --js

run:
	scala-cli --power package . --js -o main.js -f
	@echo "Serve this dir over http and open index.html, e.g.: python3 -m http.server 8000"

clean:
	scala-cli clean . 2>/dev/null || true
	find . -name ".scala-build" -type d -exec rm -rf {} +
	find . -name ".bsp" -type d -exec rm -rf {} +
	rm -f main.js main.js.map
