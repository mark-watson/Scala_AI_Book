# Practical Artificial Intelligence With Scala

This repository contains the manuscript sources and the source code for the book **Practical Artificial Intelligence With Scala** by Mark Watson.

You can read all of my books for free on my website [markwatson.com](https://markwatson.com).

If you would like to support my work, please consider purchasing my books on [Leanpub](https://leanpub.com/u/markwatson) and star my git repositories that you find useful on [GitHub](https://github.com/mark-watson?tab=repositories&q=&type=public).

---

## Repository Structure

- [source-code/](file:///Users/markwatson/GITHUB/Scala_AI_Book/source-code): Directory containing all Scala code examples.
- [manuscript/](file:///Users/markwatson/GITHUB/Scala_AI_Book/manuscript): Directory containing the manuscript drafts of the book.

---

## Projects in this Book

Each project in the [source-code/](file:///Users/markwatson/GITHUB/Scala_AI_Book/source-code) directory is a standalone [scala-cli](https://scala-cli.virtuslab.org/) project.

| Project Directory | Description |
|---|---|
| [gemini-client](file:///Users/markwatson/GITHUB/Scala_AI_Book/source-code/gemini-client) | Google Gemini REST API client (generate, search, chat, tools) |
| [openai-client](file:///Users/markwatson/GITHUB/Scala_AI_Book/source-code/openai-client) | OpenAI chat completions and embeddings client |
| [ollama-client](file:///Users/markwatson/GITHUB/Scala_AI_Book/source-code/ollama-client) | Ollama local LLM REST API client |
| [search](file:///Users/markwatson/GITHUB/Scala_AI_Book/source-code/search) | Graph, maze, and game-tree search algorithms |
| [neural-networks](file:///Users/markwatson/GITHUB/Scala_AI_Book/source-code/neural-networks) | Backpropagation neural networks (from scratch) |
| [genetic-algorithms](file:///Users/markwatson/GITHUB/Scala_AI_Book/source-code/genetic-algorithms) | Genetic algorithm optimization framework |
| [anomaly-detection](file:///Users/markwatson/GITHUB/Scala_AI_Book/source-code/anomaly-detection) | Anomaly detection using statistical methods |
| [nlp](file:///Users/markwatson/GITHUB/Scala_AI_Book/source-code/nlp) | Natural language processing: tokenizer and POS tagger |
| [semantic-web](file:///Users/markwatson/GITHUB/Scala_AI_Book/source-code/semantic-web) | Semantic Web / SPARQL with Apache Jena |
| [kgn](file:///Users/markwatson/GITHUB/Scala_AI_Book/source-code/kgn) | Knowledge Graph Navigator — queries DBPedia via SPARQL |

---

## Running the Examples

All examples are built and run using **scala-cli**. 

### 1. Install `scala-cli`
If you do not have it installed, follow the installation instructions on the [scala-cli website](https://scala-cli.virtuslab.org/install).

### 2. Run a Project
Navigate to the directory of the project you want to run, and execute:

```bash
cd source-code/<project-directory>
scala-cli run .
```

For example, to run the search examples:
```bash
cd source-code/search
scala-cli run .
```

---

## Book Cover Material, Copyright, and License

This example code is released under the **Apache License 2.0**.

Copyright 2025-2026 Mark Watson. All rights reserved.
