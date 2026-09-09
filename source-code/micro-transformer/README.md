# Tiny Transformer from Scratch — Scala AI Book

A char-level transformer with scalar autograd (embed, causal
attention, RMSNorm, MLP, cross entropy, SGD), after
`Karpathy_MicroGPT/microgpt.lisp` in the Loving Common Lisp repo.
Pure Scala, no deps, CPU only.

## Run

    scala-cli run . --main-class microtransformer.microDemo

## Test (offline)

    scala-cli run . --main-class microtransformer.transformerTest

## Book Cover Material, Copyright, and License

This example code is released using the Apache 2 license.

Copyright 2026 Mark Watson. All rights reserved.
