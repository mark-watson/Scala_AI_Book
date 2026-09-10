# Natural Language Processing

Natural Language Processing (NLP) enables computers to analyze, understand, and derive meaning from human language. In this chapter, we implement a lightweight NLP pipeline in Scala 3 from scratch, featuring a custom tokenizer, a rule-based Part-of-Speech (POS) tagger, and a named entity extractor.

All code is in `source-code/nlp`.

Dear reader, the simple approach we use here is sometimes called "a bag of words" and reflects how I did NLP work 20+ years ago. Still, this is a good example that shows simple implementations of techniques that are now made obsolete by LLMs.

## The Classic NLP Pipeline

Before large language models, NLP systems processed text in a **pipeline**: a chain of stages where each stage adds a layer of annotation the next stage can use. Raw text becomes a list of tokens, tokens get grammatical tags, tagged tokens get grouped into names and phrases, and so on up to full parsing and meaning. We build the first three stages of that pipeline here. Each stage is simple on its own, but together they turn a plain string into structured facts about who and what a sentence mentions.

It is worth knowing where this fits against modern methods. Today's transformer models learn these steps implicitly from huge corpora and rarely expose them. The classic pipeline still matters for three reasons: it is transparent, so you can see exactly why the system made a decision; it is cheap, running in milliseconds with no model to load; and it needs no training data, only dictionaries and rules. For focused tasks on a known domain, a rule-based pipeline is often the right tool, and building one teaches the structure of language that the neural black box hides.

## Tokenization

Tokenization is the process of breaking a stream of text into individual words, numbers, and punctuation marks (tokens). It looks trivial until you try it. Splitting on spaces alone mishandles "don't", "U.S.A.", "3.14", "end.", and a comma with no trailing space. A good tokenizer has to decide when a period ends a sentence and when it marks an abbreviation or a decimal point, and it has to peel trailing punctuation off a word without destroying it.

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

The design splits trailing punctuation into its own token so that "hill." becomes `hill` followed by `.`. This matters for the stages that follow: the tagger and the entity extractor want clean words, and they treat punctuation as its own signal. Note that this is **word-level** tokenization. Modern LLMs instead use **subword** tokenization (schemes like Byte Pair Encoding), which split rare words into reusable fragments so the model can handle any input with a fixed vocabulary. Word-level tokens are the natural choice here because the later stages look words up in dictionaries.

## Part-of-Speech Tagging

Part-of-Speech (POS) tagging assigns a grammatical tag (such as noun, verb, adjective, or adverb) to each token. The tags here follow the **Penn Treebank** tagset, the de facto standard in English NLP, where `NN` is a singular noun, `NNS` a plural noun, `VB` a base verb, `VBD` a past-tense verb, `VBG` a gerund, `VBN` a past participle, `DT` a determiner, `JJ` an adjective, `RB` an adverb, and `CD` a cardinal number.

Tagging is hard because words are **ambiguous**. "Rolling" can be a verb or an adjective, "down" can be a preposition or a particle, and "book" can be a noun or a verb. The right tag depends on context. Two broad approaches emerged historically: **statistical** taggers, which learn tag probabilities from a labeled corpus and pick the most likely tag sequence with an algorithm like Viterbi, and **rule-based** taggers. The most influential of the latter is the **Brill tagger** (Eric Brill, 1992), which starts by giving every word its most common tag, then applies an ordered list of transformation rules that fix tags based on the surrounding context. Our tagger is a hand-built version of exactly that idea.

We implement it in **nlp/FastTag.scala**. The tagger first loads a dictionary of word-tag mappings from `data/lexicon.txt` and performs a lookup for each token. If a word is not found, it defaults to `NN` (noun) or `NN^` (single letter):

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
```

Two choices in this first pass reflect real statistical facts about English. The lexicon stores an array of possible tags per word, and we take `tags(0)`, the most frequent tag for that word, as the starting guess. For unknown words we default to `NN`, because nouns are the largest open word class and most words a fixed dictionary misses (names, technical terms, new coinages) turn out to be nouns. This "most common tag" baseline alone is right about 90% of the time on typical text, and the rules that follow clean up the rest.

Next, it applies Brill-style transformational rules to correct tags based on suffix matching and neighboring tokens:

```scala
  ...
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

The rules draw on the two signals a tagger can cheaply exploit. Some use **context**: rule 1 says a word right after a determiner ("the") is almost certainly a noun, not a verb, so "the book" tags `book` as a noun even though `book` can be a verb. Others use **morphology**, the shape of the word itself: English suffixes strongly signal category, so `-ly` marks an adverb (rule 4), `-ed` a past participle (rule 3), `-al` an adjective (rule 5), and `-ing` a gerund. The rules run in order and each may overwrite the last, so their sequence encodes a small decision procedure. This handful of rules captures a surprising amount of English grammar for the effort.

## Named Entity Extraction

Named Entity Recognition (NER) identifies proper names, places, and organizations in text. Two approaches dominate. Statistical and neural systems (conditional random fields, then transformers) learn to spot entities from labeled examples. The older approach, which we use, combines **gazetteers**, large dictionaries of known names, with grammar rules about how name parts combine. Gazetteer methods are precise on names they know and need no training, but they miss names absent from their lists and can stumble on ambiguity, since "Washington" is both a person and a place.

In **nlp/ExtractNames.scala**, we load gazetteers (lists of first names, last names, prefix titles like "President", and place names) and check whether a run of words forms a valid name by matching it against a small grammar:

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

Each branch is a grammar rule for a name of a given length. A two-word name is a first name followed by a last name, or a title followed by a last name ("President Bush"). A three-word name adds patterns for a middle name or a title with an initial. The higher-length branches (not shown) handle middle initials, where the code checks `words(i).length == 1` and a following `"."`, so "George W. Bush" parses as one person. This is a compact **named-entity grammar** expressed directly as Scala boolean logic.

The scanning strategy matters as much as the grammar. In `getProperNames`, the extractor slides through the token list and tries the **longest match first**: it tests for a five-word name, then four, then three, and so on, advancing past a whole entity once it matches. This greedy longest-match rule (sometimes called maximal munch) is what stops "President George W. Bush" from being broken into a title plus a separate two-word name. Because it consumes the matched span before moving on, each entity is found once and at its fullest extent.

The scores and occurrences of extracted entities are compiled using a helper `ScoredList` class, which deduplicates repeated mentions and counts how often each entity appears, so a name mentioned three times is reported once with a count of three.

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

The tagger output shows the rules at work. `rolling` becomes `VBG` from its `-ing` suffix, `quickly` becomes `RB` from its `-ly` suffix, and `hill` after the determiner `the` stays a noun. The entity extractor shows longest-match in action: it reports "President George W. Bush" as a single four-token person rather than splitting it, and it correctly separates the run-together place names into London, England, Paris, France, Mexico, and Moscow. Neither stage uses any machine learning; both run on dictionaries and grammar alone, which is exactly what makes their decisions easy to inspect and adjust.
