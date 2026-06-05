# Ollama Local LLM Client — Scala AI Book

Example for Mark Watson's book "Practical Artificial Intelligence With Scala"

A REST client for the local Ollama LLM API using `requests-scala` and `ujson` in Scala. Demonstrates basic prompt completions using a local model like `mistral`.

## Setup

1. Install Ollama from [ollama.com](https://ollama.com).
2. Start the Ollama server and pull the model you want to run:

```bash
ollama pull mistral
```

## Run

    scala-cli run .

You can also override the default model name:

    scala-cli run . -- llama3

## Book Cover Material, Copyright, and License

This example code is released using the Apache 2 license.

Copyright 2025-2026 Mark Watson. All rights reserved.
