# More Hosted Model Clients — Scala AI Book

Thin REST clients for Anthropic, Mistral, Groq, Moonshot Kimi,
Perplexity, and Hugging Face inference, after the matching clients
in the Loving Common Lisp repo. Same shape as the book's Gemini and
OpenAI clients: POST JSON, parse text.

## Run (payload shapes print without keys; live calls need keys)

    scala-cli run . --main-class multiclient.clientsDemo

Set one or more of `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`,
`GROQ_API_KEY`, `MOONSHOT_API_KEY`, `PERPLEXITY_API_KEY`, `HF_API_KEY`.

## Test (offline)

    scala-cli run . --main-class multiclient.clientsTest

## Book Cover Material, Copyright, and License

This example code is released using the Apache 2 license.

Copyright 2026 Mark Watson. All rights reserved.
