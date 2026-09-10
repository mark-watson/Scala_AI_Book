# LLM Tool Use — Scala AI Book

A small tool registry (read file, list files, clock) plus an agent
loop that parses CALL and FINAL lines, after `cl-llm-agent`,
`llm-tools`, and `cl-ai-coding-agent` in the Loving Common Lisp repo.
Live loops use local Ollama when it runs.

## Run

    scala-cli run . --main-class llmtools.toolsDemo

The demo defaults to the `mistral` model. Pass a model you have pulled to
run the live loop, for example:

    scala-cli run . --main-class llmtools.toolsDemo -- qwen3.5:4b

## Test (offline, no model needed)

    scala-cli run . --main-class llmtools.toolsTest

## Book Cover Material, Copyright, and License

This example code is released using the Apache 2 license.

Copyright 2026 Mark Watson. All rights reserved.
