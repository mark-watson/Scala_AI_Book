# Genetic Algorithms

Genetic Algorithms (GAs) are optimization and search techniques inspired by the principles of natural selection and genetics. They work by evolving a population of candidate solutions (called chromosomes) over successive generations using biological operators such as selection, crossover, and mutation.

In this chapter, we build an extensible genetic algorithm framework in Scala 3 and apply it to find the maximum of a complex mathematical function.

All code is in `source-code/genetic-algorithms`.

## Evolutionary Computation and When to Reach for It

John Holland introduced genetic algorithms at the University of Michigan in the 1970s, and his 1975 book *Adaptation in Natural and Artificial Systems* set out the theory. David Goldberg's 1989 book carried the ideas to engineers and made GAs a standard tool. They belong to a wider family called **evolutionary computation**, which also includes evolution strategies and genetic programming, all built on the same loop: keep a population of candidate solutions, score them, and let the better ones produce the next generation.

The neural network chapter used gradient descent, which needs a smooth, differentiable error surface to follow downhill. Many real problems give you no such surface. The function may be jagged, discontinuous, or defined only by a black-box simulator you can call but cannot differentiate, and it may have many local optima that trap any hill-climbing method. Genetic algorithms need none of that structure. They require only a **fitness function** that scores a candidate, so they apply where calculus-based methods fail. The price is that a GA is a heuristic: it usually finds an excellent solution but carries no guarantee of the exact global optimum.

Every genetic algorithm balances two opposing forces. **Exploitation** focuses effort on the best solutions found so far, driving the population toward them. **Exploration** tries new, untested regions of the search space. Too much exploitation and the population collapses onto the first decent solution it finds, a failure called **premature convergence**. Too much exploration and the search never settles. Selection supplies the exploitation, mutation supplies the exploration, and tuning their balance is the art of applying a GA.

## Chromosomes and the Genetic Representation

A chromosome is a representation of a candidate solution. In our framework, we represent a chromosome as a fixed-length bit string using Scala's immutable `BitSet`.

It helps to keep two ideas separate. The **genotype** is the raw bit string the algorithm manipulates. The **phenotype** is what those bits mean for the actual problem, here a real number and its fitness. The genetic operators work only on the genotype and stay completely ignorant of the problem, which is what lets one engine solve many different problems.

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

The chromosome is immutable: `setBit` and `flipBit` return a new `Chromosome` rather than mutating the old one, so a parent is never disturbed when we build a child from it. The default fitness of `-999.0` acts as a sentinel meaning "not yet evaluated", and the engine resets it to this value whenever crossover or mutation changes a chromosome's bits, so a stale score can never be mistaken for a fresh one.

## The Genetic Algorithm Engine

The core GA operations are implemented in the abstract `GeneticAlgorithm` class. Users extend this class and override `calcFitness()` to define their optimization problem. This is the **template method** pattern: the base class fixes the evolutionary loop, and the subclass fills in the one problem-specific step:

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

The population starts random, spread across the whole search space so the first generation samples widely before selection begins to focus it. Seeding the random generator with a fixed value makes every run reproducible, which matters when you want to compare parameter settings.

### 1. Selection: The Roulette Wheel

To choose parents for crossover, we skew selection so that fitter chromosomes reproduce more often. A common method is **fitness-proportionate** selection, where a chromosome's chance of being picked is proportional to its raw fitness. That method has two well-known weaknesses: one chromosome with a huge fitness can dominate and cause premature convergence, and when all fitnesses are close, selection becomes almost random and progress stalls.

Our engine sidesteps both problems by selecting on **rank** rather than raw fitness. After the population is sorted best-first, we build a wheel that gives the best chromosome the most slots, the second-best one fewer, and so on down to the worst:

```scala
  // Roulette wheel: higher-ranked chromosomes get more slots
  private val rouletteWheel: Array[Int] =
    (for
      i <- 0 until populationSize
      _ <- 0 until (populationSize - i)
    yield i).toArray
```

Chromosome rank `i`$ (where 0 is the best) receives `populationSize - i`$ slots, so the number of slots falls off linearly with rank. Picking a random slot then draws a parent with a probability that depends only on its position in the ranking, never on the size of its fitness. This keeps a constant, moderate selection pressure regardless of whether the fitness values are far apart or bunched together, which makes the search far more stable than raw fitness-proportionate selection.

### 2. Crossover, Mutation, and Deduplication

The evolution of a single generation consists of calculating fitness, sorting the population, performing crossover and mutation, and deduplicating. Each step plays a distinct role, and the order matters:

```scala
  /** One generation: fitness → sort → crossover → mutate → dedup. */
  def evolve(): Unit =
    calcFitness()
    sort()
    doCrossovers()
    doMutations()
    removeDuplicates()
```

**Crossover** recombines two parents into a child, and it is the operator that does the real search. We use **single-point crossover**: pick one cut point (the `locus`), take the bits before it from the first parent and the bits after it from the second. The idea, formalized in Holland's **schema theorem**, is that good solutions are built from good partial patterns (short groups of bits called *building blocks*), and crossover spreads and combines those blocks across the population. Note where the new children go: crossover fills only the bottom `crossoverFraction` of the population, so the fittest chromosomes at the top are copied into the next generation untouched. Preserving the best members this way is called **elitism**, and it guarantees the best fitness never gets worse from one generation to the next:

```scala
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
```

**Mutation** flips a random bit in a random chromosome. On its own it is a weak search operator, but it plays a vital role: it is the only source of genetic material that selection and crossover cannot produce. Once the population loses a particular bit value at some position, only mutation can bring it back, so mutation keeps the search from getting permanently stuck. It deliberately skips index 0, again to protect the elite best chromosome:

```scala
  private def doMutations(): Unit =
    val numMutations = (populationSize * mutationFraction).toInt
    val newPop = population.toArray
    for _ <- 0 until numMutations do
      val c = rng.nextInt(populationSize - 1) + 1 // don't mutate the best (index 0)
      val g = rng.nextInt(numGenes)
      newPop(c) = newPop(c).flipBit(g).copy(fitness = -999.0)
    population = newPop.toVector
```

GAs can easily get stuck in local optima if the population loses diversity. As selection copies the best solutions, identical chromosomes tend to pile up, and a population of clones can no longer explore. To counter this, we add a **deduplication** step that flips a random bit in any chromosome that duplicates an earlier one, forcing every member to be distinct:

```scala
  private def removeDuplicates(): Unit =
    val newPop = population.toArray
    for i <- (populationSize - 1) to 4 by -1 do
      for j <- 0 until i do
        if newPop(i).bits == newPop(j).bits then
          val g = rng.nextInt(numGenes)
          newPop(i) = newPop(i).flipBit(g).copy(fitness = -999.0)
    population = newPop.toVector
```

The loop runs from the worst chromosome down to index 4, comparing each against all earlier ones, so it preserves the top few elite members and disturbs only lower-ranked duplicates. This explicit push for diversity is a direct defense against premature convergence.

## Example: Maximizing a Complex Function

We demonstrate our framework by searching for the maximum of the non-linear function:

```$
f(x) = \sin(x)\,\sin(0.4\,x)\,\sin(3\,x)
```

on the interval `[0, 10]`$. This function is a good stress test because it is **multimodal**: multiplying three sines of different frequencies produces many peaks and valleys, so a simple hill climber started from the wrong place would settle on the nearest lesser peak. We represent `x`$ using a 12-gene chromosome, giving us `2^{12} = 4096`$ possible values and a resolution of about `0.0024`$ across the interval.

The `geneToDouble` method decodes the genotype into the phenotype. It reads the bit string as a standard binary integer, then scales it into the target range:

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
    x / ((1 << numGenes) - 1).toDouble * 10.0

  private def targetFunction(x: Double): Double =
    math.sin(x) * math.sin(0.4 * x) * math.sin(3.0 * x)

  def calcFitness(): Unit =
    population = population.indices.map: i =>
      population(i).copy(fitness = targetFunction(geneToDouble(i)))
    .toVector
```

The decoding uses plain binary, where flipping a high bit can move `x`$ a long way. A common refinement is **Gray code**, an encoding in which consecutive integers differ by exactly one bit, so a single mutation makes a small change in the phenotype and the search moves more smoothly. Plain binary keeps the example simple and still works well here. Notice too the aggressive parameters chosen for this problem: an 85% crossover fraction and a high 30% mutation fraction. The heavy mutation is deliberate, keeping the small population of 20 diverse enough to escape the function's many local peaks.

The driver program runs the optimization for 20 generations:

```scala
@main def geneticAlgorithmDemo(): Unit =
  val ga = SinOptimization(numGenes = 12, popSize = 20)
  val numGenerations = 20

  for gen <- 1 to numGenerations do
    ga.evolve()
    if gen % 5 == 0 then
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

This prints the best fitness score and its location `x` as the population evolves:

```text
==================================================
Genetic Algorithm: Optimizing sin(x)*sin(0.4x)*sin(3x)
==================================================
  Generation    5:   Best fitness: 0.561433 at x=3.792430
  Generation   10:   Best fitness: 0.561702 at x=3.797314
  Generation   15:   Best fitness: 0.561790 at x=3.802198
  Generation   20:   Best fitness: 0.561790 at x=3.802198

Final population (top 5):
  Chromosome 0: fitness=0.561790  x=3.802198
  Chromosome 1: fitness=0.561790  x=3.802198
  Chromosome 2: fitness=0.561769  x=3.799756
  Chromosome 3: fitness=0.561767  x=3.804640
  Chromosome 4: fitness=0.561702  x=3.797314
```

The run shows the genetic search in action. By generation 5 the best fitness is already 0.561433, and it climbs to the global maximum of 0.561790 by generation 15 as selection and crossover combine good building blocks and mutation explores nearby values. After that the search has converged: every later generation reports the same value because the best chromosome has been found and is protected as elite, so mutation and crossover can only match it, not beat it. Through natural selection and crossover, the population converges on `x \approx 3.8022`$, which matches the true global maximum of the target function at `x \approx 3.80215`$ (where `f(x) \approx 0.561790`$) to six decimal places, without ever computing a derivative or knowing anything about the shape of the function it was optimizing.

With 12 bits the interval is divided into 4096 steps about 0.0024 wide, fine enough that the best grid point sits within about `0.0001`$ of the true peak. Coarser encodings (fewer genes) leave the answer less precise, while finer ones enlarge the search space and can make premature convergence more likely, which is the classic exploitation-versus-exploration trade-off discussed earlier.
