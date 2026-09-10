// Copyright 2025-2026 Mark Watson. All rights reserved.
//> using scala 3.6.4

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

/** Genetic algorithm engine.
  *
  * Subclass and implement `calcFitness` to define the optimization problem.
  *
  * @param numGenes            bits per chromosome
  * @param populationSize      number of chromosomes
  * @param crossoverFraction   fraction of population created by crossover
  * @param mutationFraction    fraction of population subject to mutation
  */
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

  // Roulette wheel: higher-ranked chromosomes get more slots
  private val rouletteWheel: Array[Int] =
    (for
      i <- 0 until populationSize
      _ <- 0 until (populationSize - i)
    yield i).toArray

  /** Override this to compute fitness for each chromosome. */
  def calcFitness(): Unit

  /** Sort population by fitness (descending — best first). */
  def sort(): Unit =
    population = population.sortBy(-_.fitness)

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


/** Example: find the maximum of sin(x) * sin(0.4x) * sin(3x) on [0, 10]. */
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

  def printBest(): Unit =
    val x = geneToDouble(0)
    println(f"  Best fitness: ${population(0).fitness}%.6f at x=$x%.6f")

@main def geneticAlgorithmDemo(): Unit =
  println("=" * 50)
  println("Genetic Algorithm: Optimizing sin(x)*sin(0.4x)*sin(3x)")
  println("=" * 50)

  val ga = SinOptimization(numGenes = 12, popSize = 20)
  val numGenerations = 20

  for gen <- 1 to numGenerations do
    ga.evolve()
    if gen % 5 == 0 then
      print(f"  Generation $gen%4d: ")
      ga.calcFitness()
      ga.sort()
      ga.printBest()

  println("\nFinal population (top 5):")
  ga.calcFitness()
  ga.sort()
  for i <- 0 until 5 do
    val x = ga.geneToDouble(i)
    println(f"  Chromosome $i: fitness=${ga.population(i).fitness}%.6f  x=$x%.6f")
