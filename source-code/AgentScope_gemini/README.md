# AgentScope with Google Gemini — Scala Example

Book URI: https://leanpub.com/scalaai

Demonstrates the [AgentScope](https://java.agentscope.io) multi-agent framework with Google's **Gemini 2.5 Flash** model from Scala 3. The `ReActAgent` implements a Reasoning + Acting loop: the agent reasons about a user message, optionally invokes tools, and returns a final response.

## Prerequisites

- Scala CLI 1.14+ ([install](https://scala-cli.virtuslab.org/install))
- A valid Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

## Setup

```bash
export GOOGLE_API_KEY=your_key_here
```

## Run

```bash
# Simple ReActAgent demo
scala-cli run . --main-class agentscope_gemini.agentScopeDemo

# Tool-use demo (weather stub + filesystem tools)
scala-cli run . --main-class agentscope_gemini.toolUseDemo
```

## Key Dependencies

Declared via `//> using dep` directives in `GeminiConfig.scala`:

| Artifact | Purpose |
|---|---|
| `io.agentscope:agentscope:1.0.12` | AgentScope core (agents, messaging, tools) |
| `com.google.genai:google-genai:1.59.0` | Google GenAI SDK (Gemini models) |
| `org.slf4j:slf4j-simple:2.0.18` | SLF4J logging |

## Book Cover Material, Copyright, and License

This example is released using the Apache 2 license.

Copyright 2022-2026 Mark Watson. All rights reserved.

## This Book is Licensed with Creative Commons Attribution CC BY Version 3

You are free to share and adapt this content, with attribution.
