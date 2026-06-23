# Genetic Algorithms

Genetic Algorithms (GAs) are optimization and search techniques inspired by the principles of natural selection and genetics. They work by evolving a population of candidate solutions (called chromosomes) over successive generations using biological operators such as selection, crossover, and mutation.

In this chapter, we build an extensible genetic algorithm framework in Scala 3 and apply it to find the maximum of a complex mathematical function.

All code is in `source-code/genetic-algorithms`.

## Chromosomes and the Genetic Representation

A chromosome is a representation of a candidate solution. In our framework, we represent a chromosome as a fixed-length bit string using Scala's immutable `BitSet`. 

We define the chromosome and operations to query, set, or flip bits in **genetic-algorithms/GeneticAlgorithm.scala**:

```scala
import scala.collection.immutable.BitSet
import scala.util.Random

/** A chromosome is a fixed-length bit string with an associated fitness score. */
case class Chromosome(bits: BitSet, numGenes: Int, fitness: Double = -999.0):
  def getBit(index: Int): Boolean = bits.contains(index)

  def setBit(index: Int, value: Boolean): Chromosome =
    if value then copy(bits = bits + index)
    else copy(bits = bits - index)

  def flipBit(index: Int): Chromosome =
    if bits.contains(index) then copy(bits = bits - index)
    else copy(bits = bits + index)

  override def toString: String =
    val bitStr = (0 until numGenes).map(i => if getBit(i) then '1' else '0').mkString
    f"Chromosome[$bitStr, fitness=$fitness%.6f]"

object Chromosome:
  /** Create a chromosome with random bits. */
  def random(numGenes: Int, rng: Random): Chromosome =
    val bits = (0 until numGenes).filter(_ => rng.nextBoolean()).to(BitSet)
    Chromosome(bits, numGenes)
```

## The Genetic Algorithm Engine

The core GA operations are implemented in the abstract `GeneticAlgorithm` class. Users extend this class and override `calcFitness()` to define their optimization problem:

```scala
abstract class GeneticAlgorithm(
    val numGenes: Int,
    val populationSize: Int,
    val crossoverFraction: Double = 0.8,
    val mutationFraction: Double = 0.01,
    seed: Long = 42L
):
  protected val rng: Random = Random(seed)

  var population: Vector[Chromosome] =
    Vector.fill(populationSize)(Chromosome.random(numGenes, rng))
```

### 1. Selection: The Roulette Wheel

To choose parents for crossover, we use a **roulette wheel selection** method. Rather than selecting parents at random, we skew selection so that chromosomes with higher fitness scores have a higher probability of reproducing. 

We initialize a pre-indexed selection wheel:

```scala
  // Roulette wheel: higher-ranked chromosomes get more slots
  private val rouletteWheel: Array[Int] =
    (for
      i <- 0 until populationSize
      _ <- 0 until (populationSize - i)
    yield i).toArray
```

### 2. Crossover, Mutation, and Deduplication

The evolution of a single generation consists of calculating fitness, sorting the population, performing crossover and mutation, and deduplicating. GAs can easily get stuck in local optima if the population loses diversity. To counter this, we implement a deduplication step that flips a random bit in duplicate chromosomes:

```scala
  /** One generation: fitness → sort → crossover → mutate → dedup. */
  def evolve(): Unit =
    calcFitness()
    sort()
    doCrossovers()
    doMutations()
    removeDuplicates()

  private def doCrossovers(): Unit =
    val numCrossovers = (populationSize * crossoverFraction).toInt
    val newPop = population.toArray
    val startIndex = populationSize - numCrossovers
    for i <- startIndex until populationSize do
      val c1 = rouletteWheel(rng.nextInt(rouletteWheel.length))
      val c2 = rouletteWheel(rng.nextInt(rouletteWheel.length))
      if c1 != c2 then
        val locus = rng.nextInt(numGenes - 2) + 1
        var child = newPop(i).copy(fitness = -999.0)
        for g <- 0 until numGenes do
          val parent = if g < locus then population(c1) else population(c2)
          child = child.setBit(g, parent.getBit(g))
        newPop(i) = child
    population = newPop.toVector

  private def doMutations(): Unit =
    val numMutations = (populationSize * mutationFraction).toInt
    val newPop = population.toArray
    for _ <- 0 until numMutations do
      val c = rng.nextInt(populationSize - 1) + 1 // don't mutate the best (index 0)
      val g = rng.nextInt(numGenes)
      newPop(c) = newPop(c).flipBit(g).copy(fitness = -999.0)
    population = newPop.toVector

  private def removeDuplicates(): Unit =
    val newPop = population.toArray
    for i <- (populationSize - 1) to 4 by -1 do
      for j <- 0 until i do
        if newPop(i).bits == newPop(j).bits then
          val g = rng.nextInt(numGenes)
          newPop(i) = newPop(i).flipBit(g).copy(fitness = -999.0)
    population = newPop.toVector
```

## Example: Maximizing a Complex Function

We demonstrate our framework by searching for the maximum of the non-linear function:

{$$}
f(x) = \sin(x) \sin(0.4x) \sin(3x)
{/$$}

on the interval {$$}[0, 10]{/$$}. We represent {$$}x{/$$} using a 10-gene chromosome, giving us {$$}2^{10} = 1024{/$$} possible values.

We implement the optimization in `SinOptimization`:

```scala
class SinOptimization(numGenes: Int, popSize: Int)
    extends GeneticAlgorithm(numGenes, popSize, crossoverFraction = 0.85, mutationFraction = 0.3):

  /** Convert a chromosome's bit pattern to a Double in [0, 10]. */
  def geneToDouble(index: Int): Double =
    var x = 0.0
    var base = 1.0
    for j <- 0 until numGenes do
      if population(index).getBit(j) then x += base
      base *= 2
    x / 102.4

  private def targetFunction(x: Double): Double =
    math.sin(x) * math.sin(0.4 * x) * math.sin(3.0 * x)

  def calcFitness(): Unit =
    population = population.indices.map: i =>
      population(i).copy(fitness = targetFunction(geneToDouble(i)))
    .toVector
```

The driver program runs the optimization for 500 generations:

```scala
@main def geneticAlgorithmDemo(): Unit =
  val ga = SinOptimization(numGenes = 10, popSize = 20)
  val numGenerations = 500

  for gen <- 0 until numGenerations do
    ga.evolve()
    if gen % 50 == 0 || gen == numGenerations - 1 then
      print(f"  Generation $gen%4d: ")
      ga.calcFitness()
      ga.sort()
      ga.printBest()
```

## Running the Optimization

Run the project with:

```bash
scala-cli run .
```

This prints the best fitness score and its location {$$}x{/$$} as the population evolves:

```text
==================================================
Genetic Algorithm: Optimizing sin(x)*sin(0.4x)*sin(3x)
==================================================
  Generation    0:   Best fitness: 0.820253 at x=4.199219
  Generation   50:   Best fitness: 0.903823 at x=5.664062
  Generation  100:   Best fitness: 0.904803 at x=5.761719
  ...
  Generation  450:   Best fitness: 0.904807 at x=5.751953
  Generation  499:   Best fitness: 0.904807 at x=5.751953

Final population (top 5):
  Chromosome 0: fitness=0.904807  x=5.751953
  Chromosome 1: fitness=0.904807  x=5.751953
  Chromosome 2: fitness=0.904807  x=5.751953
  Chromosome 3: fitness=0.904807  x=5.751953
  Chromosome 4: fitness=0.904586  x=5.771484
```

Through natural selection and crossover, the population quickly converges on {$$}x \approx 5.75{/$$}, which is the global maximum of the target function in the range {$$}[0, 10]{/$$}.
