// Copyright 2025-2026 Mark Watson. All rights reserved.

package nlp

@main def nlpDemo(args: String*): Unit =
  println("=" * 50)
  println("Natural Language Processing (NLP) in Scala")
  println("=" * 50)

  // 1. Test Tokenizer & POS Tagger
  println("\n--- 1. Part-of-Speech Tagging ---")
  val sampleText = "The ball, rolling quickly, went down the hill."
  println(s"Input Text: '$sampleText'")

  val tokens = Tokenizer.wordsToList(sampleText)
  val tagger = FastTag()
  val tags = tagger.tag(tokens)

  print("POS Tags:   ")
  for (word, tag) <- tokens.zip(tags) do
    print(s"$word/$tag ")
  println()

  // 2. Test Name and Place Extraction
  println("\n--- 2. Entity Extraction ---")
  val names = ExtractNames()

  println(s"Los Angeles is place:        ${names.isPlaceName("Los Angeles")}")
  println(s"President Bush is human:     ${names.isHumanName("President Bush")}")
  println(s"George W. Bush is human:     ${names.isHumanName("George W. Bush")}")

  val entityText = "George Bush played golf. President George W. Bush went to London England, " +
    "Paris France and Mexico to see Mary Smith in Moscow. President Bush will return home Monday."
  println(s"\nExtracting entities from:\n'$entityText'")

  val (humanNames, placeNames) = names.getProperNames(entityText)
  println(s"\nHuman names extracted: ${humanNames.getValuesAsString}")
  println(s"Place names extracted: ${placeNames.getValuesAsString}")
