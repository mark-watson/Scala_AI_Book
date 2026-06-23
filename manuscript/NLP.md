# Natural Language Processing

Natural Language Processing (NLP) enables computers to analyze, understand, and derive meaning from human language. In this chapter, we implement a lightweight NLP pipeline in Scala 3 from scratch, featuring a custom tokenizer, a rule-based Part-of-Speech (POS) tagger, and a named entity extractor.

All code is in `source-code/nlp`.

## Tokenization

Tokenization is the process of breaking a stream of text into individual words, numbers, and punctuation marks (tokens). 

In **nlp/Tokenizer.scala**, we implement a tokenizer using Java's standard `StreamTokenizer`, cleaning control characters and splitting trailing punctuation into separate tokens:

```scala
object Tokenizer:

  def wordsToList(s2: String): List[String] =
    val clean = stripControlCharacters(s2)
    val words = ListBuffer[String]()
    var count = 0
    try
      val strTok = StreamTokenizer(StringReader(clean))
      strTok.whitespaceChars('"', '"')
      strTok.whitespaceChars('\'', '\'')
      strTok.whitespaceChars('/', '/')

      while strTok.nextToken() != StreamTokenizer.TT_EOF do
        val s = strTok.ttype match
          case StreamTokenizer.TT_EOL => ""
          case StreamTokenizer.TT_WORD => strTok.sval
          case StreamTokenizer.TT_NUMBER => strTok.nval.toInt.toString
          case ttype => ttype.toChar.toString

        if s.nonEmpty then
          if s.endsWith(".") then
            words.append(s.substring(0, s.length - 1))
            words.append(".")
          else if s.endsWith(",") then
            val x = s.substring(0, s.length - 1)
            if x.nonEmpty then words.append(x)
            words.append(",")
          ...
          else
            words.append(s)
    catch
      case e: IOException => e.printStackTrace()
    words.toList
```

## Part-of-Speech Tagging

Part-of-Speech (POS) tagging assigns a grammatical tag (such as noun, verb, adjective, or adverb) to each token based on its definition and context.

We implement this in **nlp/FastTag.scala**. The tagger first loads a dictionary of word-tag mappings from `data/lexicon.txt` and performs a lookup for each token. If a word is not found, it defaults to `NN` (noun) or `NN^` (single letter). 

Next, it applies Brill-style transformational rules to correct tags based on suffix matching and neighboring tokens:

```scala
class FastTag(dataPath: String = "data/lexicon.txt"):
  private val lexicon = collection.mutable.Map[String, Array[String]]()
  ...

  def tag(words: List[String]): List[String] =
    val ret = collection.mutable.ListBuffer[String]()
    for word <- words do
      val ss = lexicon.get(word).orElse(lexicon.get(word.toLowerCase))
      ss match
        case Some(tags) => ret.append(tags(0))
        case None =>
          if word.length == 1 then ret.append(word + "^")
          else ret.append("NN")

    val result = ret.toArray
    // Apply transformational rules
    for i <- words.indices do
      val word = result(i)
      
      // Rule 1: DT, {VBD | VBP | VB} --> DT, NN
      if i > 0 && result(i - 1) == "DT" then
        if word == "VBD" || word == "VBP" || word == "VB" then
          result(i) = "NN"

      // Rule 2: Convert a noun to a number (CD) if it contains "." or parses as float
      if result(i).startsWith("N") && (words(i).contains(".") || words(i).toFloatOption.isDefined) then
        result(i) = "CD"

      // Rule 3: Convert a noun to a past participle if it ends in "ed"
      if result(i).startsWith("N") && words(i).endsWith("ed") then
        result(i) = "VBN"

      // Rule 4: Convert any type to adverb if it ends in "ly"
      if words(i).endsWith("ly") then
        result(i) = "RB"

      // Rule 5: Convert common noun to adjective if it ends in "al"
      if result(i).startsWith("NN") && words(i).endsWith("al") then
        result(i) = "JJ"

      // Rule 6: Convert noun to verb if preceding word is "would"
      if i > 0 && result(i).startsWith("NN") && words(i - 1).equalsIgnoreCase("would") then
        result(i) = "VB"
      ...
    result.toList
```

## Named Entity Extraction

Named Entity Recognition (NER) identifies proper names, places, and organizations in text.

In **nlp/ExtractNames.scala**, we implement a rule-based name and place extractor. The class loads gazetteers (large lists of first names, last names, prefix titles like "President", and place names) and scans the text matching n-grams (from 1-grams to 5-grams) to extract human and place names:

```scala
class ExtractNames(dataDir: String = "data/"):
  private val firstNameSet = collection.mutable.Set[String]()
  private val lastNameSet = collection.mutable.Set[String]()
  private val prefixSet = collection.mutable.Set[String]()
  private val placeNameMap = collection.mutable.Map[String, String]()
  ...

  def isHumanName(words: List[String]): Boolean =
    val len = words.size
    if len == 1 then
      lastNameSet.contains(words(0))
    else if len == 2 then
      (firstNameSet.contains(words(0)) && lastNameSet.contains(words(1))) ||
      (prefixSet.contains(words(0)) && lastNameSet.contains(words(1)))
    else if len == 3 then
      (firstNameSet.contains(words(0)) && firstNameSet.contains(words(1)) && lastNameSet.contains(words(2))) ||
      (prefixSet.contains(words(0)) && words(1) == "." && lastNameSet.contains(words(2)))
    ...
```

The scores and occurrences of extracted entities are compiled using a helper `ScoredList` class.

## Running the NLP Demo

The entry point in **nlp/Main.scala** processes sample sentences through both POS tagging and entity extraction:

```scala
@main def nlpDemo(args: String*): Unit =
  // 1. Test POS Tagger
  val sampleText = "The ball, rolling quickly, went down the hill."
  val tokens = Tokenizer.wordsToList(sampleText)
  val tagger = FastTag()
  val tags = tagger.tag(tokens)

  print("POS Tags:   ")
  for (word, tag) <- tokens.zip(tags) do
    print(s"$word/$tag ")
  println()

  // 2. Test Named Entity Extraction
  val names = ExtractNames()
  val entityText = "George Bush played golf. President George W. Bush went to London England, " +
    "Paris France and Mexico to see Mary Smith in Moscow."
  val (humanNames, placeNames) = names.getProperNames(entityText)
  println(s"Human names extracted: ${humanNames.getValuesAsString}")
  println(s"Place names extracted: ${placeNames.getValuesAsString}")
```

Running the pipeline using `scala-cli run .` outputs:

```text
==================================================
Natural Language Processing (NLP) in Scala
==================================================

--- 1. Part-of-Speech Tagging ---
Input Text: 'The ball, rolling quickly, went down the hill.'
POS Tags:   The/DT ball/NN ,/, rolling/VBG quickly/RB ,/, went/VBD down/RP the/DT hill/NN ./. 

--- 2. Entity Extraction ---
Los Angeles is place:        true
President Bush is human:     true
George W. Bush is human:     true

Extracting entities from:
'George Bush played golf. President George W. Bush went to London England, Paris France and Mexico to see Mary Smith in Moscow. President Bush will return home Monday.'

Human names extracted: President George W. Bush:1, George Bush:1, Mary Smith:1, President Bush:1
Place names extracted: London:1, England:1, Paris:1, France:1, Mexico:1, Moscow:1
```

The POS tagger successfully handles verbs and adverbs (e.g. `rolling/VBG`, `quickly/RB`), and the entity extractor matches the complex prefixes, middle initials, and locations.
