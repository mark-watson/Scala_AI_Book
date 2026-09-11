//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import java.util.concurrent.{CompletableFuture, ConcurrentHashMap, ConcurrentLinkedQueue, CountDownLatch, TimeUnit}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger, AtomicReference, DoubleAdder}
import scala.collection.mutable
import scala.concurrent.duration.{Duration, FiniteDuration}

// ============================================================================
// Search.scala -- Monte Carlo Tree Search.
//
// Implements the techniques from section 2 of the design document:
//
//   2.1  MCTS with UCT / PUCT selection
//   2.2  policy-network priors instead of random playouts (optionally)
//   2.3  RAVE / AMAF value blending
//   2.4  progressive widening of the move list
//   2.5  parallel search on virtual threads with virtual loss
//   2.6  (alpha-beta tactical supplement -- see the note at the end of the file)
//
// One MCTS iteration is: select -> expand -> evaluate -> backpropagate.
//
// Threading model
//   The tree is shared and mutable; positions are immutable.  Nodes hold
//   atomic counters, so any number of virtual threads can descend the same
//   tree at once.  Threads are kept apart by *virtual loss*: while a thread is
//   working on a branch it temporarily depresses that branch's value, so its
//   peers pick somewhere else.  This is what makes the parallel search explore
//   rather than pile onto one line.
//
// Value convention
//   Each node stores Q from the perspective of the player who *made the move
//   into that node*.  `PolicyValueNetwork.evaluate` returns a value from the
//   perspective of the player to move, so search negates it as it backs up.
// ============================================================================

/** How leaf positions are scored. */
enum LeafEvaluation:
  /** Use the network's value head. */
  case ValueHead
  /** Play a random game to the end and use its result (design doc section 2.1). */
  case Rollout
  /**
   * Use the value head when the network's value carries information, and
   * random rollouts when it does not (e.g. [[RandomNetwork]]).
   */
  case Auto

/**
 * Everything tunable about a search.
 *
 * The defaults are chosen to be reasonable on a 9x9 board and to keep a single
 * move responsive in interactive play.
 */
final case class SearchConfig(
    /** Maximum MCTS iterations.  The real count can be lower if time runs out. */
    playouts: Int = 1600,
    /** Worker count.  Virtual threads make a large number cheap. */
    threads: Int = math.max(1, Runtime.getRuntime.availableProcessors()),
    /** Hard wall-clock budget for this search, if any. */
    timeLimit: Option[FiniteDuration] = None,
    /** Exploration constant in the PUCT formula. */
    c_puct: Double = 1.5,
    /** Temporary value penalty applied to a branch while a thread explores it. */
    virtualLoss: Double = 1.0,
    /** Blend in RAVE/AMAF statistics (design doc section 2.3). */
    raveEnabled: Boolean = true,
    /** `k` in the RAVE mixing formula `beta = sqrt(k / (3N + k))`. */
    raveEquivalence: Double = 3000.0,
    /**
     * How many of the rollout's earliest moves the AMAF update considers.
     * Full AMAF costs `O(path x playout)` and quickly outweighs its benefit.
     */
    raveHorizon: Int = 30,
    /** Limit how many children a node may have, based on its visit count. */
    progressiveWidening: Boolean = true,
    /** `C` in `children = C * N^alpha`. */
    wideningC: Double = 2.0,
    /** `alpha` in `children = C * N^alpha`. */
    wideningAlpha: Double = 0.5,
    evaluation: LeafEvaluation = LeafEvaluation.Auto,
    /** Depth cap for random rollouts, so a single iteration stays bounded. */
    maxRolloutDepth: Int = 250,
    /** Dirichlet noise concentration for root exploration during self-play. */
    dirichletAlpha: Double = 0.3,
    /** Mixing weight of the root noise; 0 disables it. */
    dirichletWeight: Double = 0.0,
    /** Batch size for network evaluation; > 1 helps the ONNX/CoreML path. */
    maxBatchSize: Int = 1,
    seed: Long = 0x5eedL
):
  require(playouts > 0, "playouts must be positive")
  require(threads > 0, "threads must be positive")
  require(progressiveWidening && wideningC > 0, "wideningC must be positive")
  require(wideningAlpha > 0 && wideningAlpha <= 1, "wideningAlpha must be in (0, 1]")
  require(virtualLoss >= 0, "virtualLoss must not be negative")

object SearchConfig:
  /** A fast configuration for interactive play. */
  def interactive(size: Int): SearchConfig = SearchConfig(
    playouts = if size <= 9 then 2000 else 4000,
    threads = math.max(1, Runtime.getRuntime.availableProcessors()),
    timeLimit = Some(FiniteDuration(3, "s"))
  )

  /** A strong-but-slow configuration, as used for self-play games. */
  def selfPlay(size: Int): SearchConfig = SearchConfig(
    playouts = if size <= 9 then 800 else 1200,
    threads = math.max(1, Runtime.getRuntime.availableProcessors()),
    dirichletAlpha = 0.3,
    dirichletWeight = 0.25
  )

/**
 * One node of the search tree.
 *
 * Children are created lazily and stored in a concurrent map, which is what
 * lets several virtual threads widen the same node at once.  Values are
 * accumulated with `DoubleAdder`, the JDK's lock-free double accumulator; the
 * design document calls this field `AtomicDouble`, which is not in the JDK.
 */
final class MCTSNode(
    /** The move that led to this node.  For the root this is `Point.Pass`. */
    val move: Point,
    /** `P(move)` from the policy network. */
    val prior: Float,
    val parent: MCTSNode | Null,
    /** The position *after* `move`. */
    val state: BoardState
):
  private[go] val childMap = new ConcurrentHashMap[Int, MCTSNode]()

  /** `N` -- visit count. */
  val visits = new AtomicInteger(0)

  /** Total of values seen, in the units described at the top of this file. */
  val totalValue = new DoubleAdder()

  /** RAVE `N` -- how often this move appeared anywhere in a playout. */
  val raveVisits = new AtomicInteger(0)

  /** RAVE total value. */
  val raveTotalValue = new DoubleAdder()

  /** Legal moves and their priors, ordered by descending prior. */
  private val candidateRef = new AtomicReference[Array[Int]](null)
  private val priorRef = new AtomicReference[Array[Float]](null)

  def isRoot: Boolean = parent == null

  /** True once the policy network has been evaluated at this node. */
  def isEvaluated: Boolean = candidateRef.get() != null

  def candidates: Array[Int] = candidateRef.get()

  def candidatePriors: Array[Float] = priorRef.get()

  def n: Int = visits.get()

  /** Mean value from the perspective of the player who made this move. */
  def q: Double =
    val v = visits.get()
    if v == 0 then 0.0 else totalValue.sum() / v

  def raveQ: Double =
    val v = raveVisits.get()
    if v == 0 then 0.0 else raveTotalValue.sum() / v

  def children: Array[MCTSNode] = childMap.values().toArray(new Array[MCTSNode](0))

  def childCount: Int = childMap.size()

  def childFor(moveIndex: Int): MCTSNode | Null = childMap.get(moveIndex)

  /**
   * Records the network's policy at this node and orders the legal moves by
   * prior.  Safe to call from several threads: only the first installation
   * wins, and the others keep their (identical) values.
   *
   * Two rules keep the tree honest and make games finish:
   *
   *   - moves that fill the mover's own eye are dropped, because they can only
   *     destroy the group they belong to;
   *   - when nothing but own-eye fills remain, passing is the only candidate,
   *     so the engine stops playing instead of dismantling its own position.
   *
   * Priors are renormalised after filtering, so root noise and PUCT always see
   * a distribution that sums to one.
   *
   * Returns true when this call was the one that installed the candidates.
   */
  private[go] def install(policy: Array[Float]): Boolean =
    if candidateRef.get() != null then false
    else
      val n = state.area
      val mover = state.toMove
      val legal = state.legalMoves(mover)
      val useful = legal.filterNot(p => state.isOwnEye(p, mover))
      val (indices, priors) =
        if useful.isEmpty then (Vector(n), Vector(1f))
        else
          // A floor on the pass prior keeps the option to stop explorable, and
          // a floor on move priors keeps a move the network dislikes reachable.
          val moveIndices = useful.map(_.index)
          val movePriors = useful.map(p => math.max(1e-6f, policy(p.index)))
          val passPrior = math.max(0.02f, policy(n))
          (moveIndices :+ n, movePriors :+ passPrior)

      val order = indices.indices.sortBy(i => -priors(i).toDouble)
      val orderedIndices = order.map(indices).toArray
      val orderedPriorsRaw = order.map(priors).toArray
      val total = orderedPriorsRaw.sum
      val orderedPriors =
        if total > 0f then orderedPriorsRaw.map(_ / total) else orderedPriorsRaw

      if candidateRef.compareAndSet(null, orderedIndices) then
        priorRef.set(orderedPriors)
        true
      else false

  override def toString: String =
    f"${move.label(state.size)} n=$n q=$q%.3f prior=$prior%.3f"

/**
 * Plays a light random game from a position and returns the outcome from the
 * perspective of the player to move at the start.
 *
 * Rollouts are the original source of MCTS strength and remain the fallback
 * when no trained value network is available.  Two details keep them sane:
 * moves that fill the mover's own eye are never played, so groups are not
 * casually destroyed, and the length is capped.
 */
object LightPlayout:
  /**
   * Plays a light random game and returns `(value, movesPlayed)`.
   *
   * `value` is from the perspective of the player to move at the start, and
   * `movesPlayed` holds flat point indices in the order played, which is what
   * the RAVE update needs.  Moves that fill the mover's own eye are skipped:
   * without that rule a random rollout cheerfully destroys its own groups.
   */
  def playout(start: BoardState, maxDepth: Int, rng: java.util.Random): (Float, Array[Int]) =
    val firstPlayer = start.toMove
    val played = new Array[Int](maxDepth)
    var playedCount = 0
    var s = start
    var depth = 0
    while !s.isTerminal && depth < maxDepth do
      val color = s.toMove
      val legal = s.legalMoves(color)
      val chosen =
        if legal.isEmpty then Point.Pass
        else
          val playable = legal.filterNot(p => s.isOwnEye(p, color))
          if playable.isEmpty then Point.Pass
          else playable(rng.nextInt(playable.length))
      s = s.place(chosen, color).fold(_ => s.playPass, identity)
      played(playedCount) = chosen.index
      playedCount += 1
      depth += 1
    (Search.terminalValueFrom(s, firstPlayer), java.util.Arrays.copyOf(played, playedCount))

  /** Convenience wrapper for callers that only want the value. */
  def playoutValue(start: BoardState, maxDepth: Int, rng: java.util.Random): Float =
    playout(start, maxDepth, rng)._1

/**
 * Collects leaf evaluations from many virtual threads and feeds them to the
 * network in batches.
 *
 * A GPU is only faster than a CPU when work arrives in batches, so the ONNX
 * path benefits enormously from this.  Callers block on a future; the batcher
 * thread gathers whatever is queued (up to `maxBatchSize`) and issues one
 * `evaluateBatch` call.
 *
 * Robustness matters more than throughput here: every caller has a timeout and
 * falls back to evaluating on its own thread, and `close()` drains whatever is
 * still queued, so a search can never hang on a lost request.
 */
private[go] final class BatchingEvaluator(
    network: PolicyValueNetwork,
    maxBatchSize: Int,
    collectTimeoutNanos: Long = 2_000_000L
) extends AutoCloseable:
  private final case class Request(state: BoardState, promise: CompletableFuture[(Array[Float], Float)])

  private val queue = new ConcurrentLinkedQueue[Request]()
  private val running = new AtomicBoolean(true)
  private val batcher = Thread
    .ofPlatform()
    .name("go-eval-batcher")
    .daemon(true)
    .start(() => loop())

  def evaluate(state: BoardState): (Array[Float], Float) =
    if maxBatchSize <= 1 || !running.get() then network.evaluate(state)
    else
      val promise = new CompletableFuture[(Array[Float], Float)]()
      queue.add(Request(state, promise))
      try promise.get(2, TimeUnit.SECONDS)
      catch case _: Throwable => network.evaluate(state)

  private def loop(): Unit =
    while running.get() do
      var first = queue.poll()
      if first == null then
        // Brief nap so an idle batcher does not spin a core.
        try Thread.sleep(collectTimeoutNanos / 1_000_000L, (collectTimeoutNanos % 1_000_000L).toInt)
        catch case _: InterruptedException => ()
        first = queue.poll()
      if first != null then
        val batch = mutable.ArrayBuffer(first)
        var next = queue.poll()
        while next != null && batch.length < maxBatchSize do
          batch += next
          next = queue.poll()
        try
          val results = network.evaluateBatch(batch.map(_.state).toSeq)
          batch.zip(results).foreach((req, res) => req.promise.complete(res))
        catch
          case e: Throwable => batch.foreach(req => req.promise.completeExceptionally(e))

  override def close(): Unit =
    running.set(false)
    var pending = queue.poll()
    while pending != null do
      try pending.promise.complete(network.evaluate(pending.state))
      catch case e: Throwable => pending.promise.completeExceptionally(e)
      pending = queue.poll()

/** The result of one [[Search.search]] call. */
final case class SearchResult(
    move: Point,
    /** Normalised visit distribution over the root's children -- the improved policy. */
    policy: Map[Point, Float],
    /** Raw visit counts, useful for inspection and for tests. */
    visits: Map[Point, Int],
    /** Playouts actually completed. */
    playouts: Int,
    elapsed: FiniteDuration,
    /** Root win rate from the perspective of the player to move. */
    winRate: Double,
    /** Most-visited line from the root. */
    principalVariation: Vector[Point],
    /** Prior the network gave to the chosen move, before search. */
    bestPrior: Float,
    nodesCreated: Int
):
  override def toString: String =
    val line = principalVariation.map(p => p.toGtp(19)).mkString(" ")
    f"move=${move.toGtp(19)}%s winRate=$winRate%.3f playouts=$playouts pv=$line%s"

/**
 * A search tree that can be extended over many calls.
 *
 * Keeping the tree alive is what makes pondering possible: the engine can keep
 * searching while the opponent thinks, then reuse the part of the tree that is
 * still relevant.
 */
final class SearchSession(
    val network: PolicyValueNetwork,
    val config: SearchConfig,
    val root: MCTSNode
):
  import Search.*

  private val batcher: BatchingEvaluator | Null =
    if config.maxBatchSize > 1 then new BatchingEvaluator(network, config.maxBatchSize) else null

  private val nodesCreated = new AtomicInteger(1)

  /** Total playouts completed across every call to [[runPlayouts]]. */
  private val completedPlayouts = new AtomicInteger(0)

  /** True when rollouts, rather than the value head, are being used. */
  val usesRollouts: Boolean = config.evaluation match
    case LeafEvaluation.Rollout                                        => true
    case LeafEvaluation.ValueHead                                      => false
    case LeafEvaluation.Auto                                           => !network.valueIsMeaningful

  private def evaluate(state: BoardState): (Array[Float], Float) =
    if batcher != null then batcher.evaluate(state) else network.evaluate(state)

  /**
   * Mixes Dirichlet noise into the root priors, as AlphaGo Zero does, so that
   * self-play games do not all follow the same line.  Must be called before
   * any parallel playouts start.
   */
  def applyRootNoise(rng: java.util.Random): Unit =
    if config.dirichletWeight > 0 then
      ensureRootEvaluated(rng)
      val priors = root.candidatePriors
      if priors != null && priors.nonEmpty then
        val noise = dirichlet(priors.length, config.dirichletAlpha, rng)
        var i = 0
        while i < priors.length do
          priors(i) = ((1.0 - config.dirichletWeight) * priors(i) +
            config.dirichletWeight * noise(i)).toFloat
          i += 1

  /** Evaluates the root if it has not been evaluated yet. */
  private def ensureRootEvaluated(rng: java.util.Random): Unit =
    if !root.isEvaluated then
      val (policy, _) = evaluate(root.state)
      root.install(policy)

  /**
   * Runs `playouts` more iterations, in parallel.  Returns the number actually
   * completed, which is lower if a time limit stopped the search.
   */
  def runPlayouts(playouts: Int, deadlineNanos: Option[Long] = None): Int =
    ensureRootEvaluated(new java.util.Random(config.seed))
    val threadCount = math.max(1, config.threads)
    val budget = new AtomicInteger(playouts)
    val latch = new CountDownLatch(threadCount)
    val completed = new AtomicInteger(0)

    var t = 0
    while t < threadCount do
      val rng = new java.util.Random(config.seed + t * 7919L)
      Thread.ofVirtual().name(s"mcts-$t").start(() =>
        try
          var running = true
          while running do
            if deadlineNanos.exists(System.nanoTime() > _) then running = false
            else if budget.getAndDecrement() <= 0 then running = false
            else
              iterate(rng)
              completed.incrementAndGet()
        finally latch.countDown()
      )
      t += 1

    latch.await()
    if batcher != null then batcher.close()
    val done = completed.get()
    completedPlayouts.addAndGet(done)
    done

  /**
   * One MCTS iteration: select a path, expand the leaf, evaluate it, and back
   * the value up, applying virtual loss along the way.
   *
   * Virtual loss works like this: descending applies `visits += 1` and
   * `value -= virtualLoss` to every node on the path, and backpropagation then
   * adds `realValue + virtualLoss`.  The two cancel exactly, so each iteration
   * contributes one visit and one value, while in-flight iterations are
   * simultaneously discounted and therefore avoided by the other threads.
   */
  private def iterate(rng: java.util.Random): Unit =
    var node: MCTSNode = root
    val path = mutable.ArrayBuffer[MCTSNode](node)
    applyVirtualLoss(node)
    var value = 0.0f
    var rolloutMoves: Array[Int] = null
    var descending = true

    while descending do
      if node.state.isTerminal then
        value = terminalValue(node.state)
        descending = false
      else if !node.isEvaluated then
        // A fresh leaf: ask the network for both a policy and a value.
        val (policy, v) = evaluate(node.state)
        if node.install(policy) then nodesCreated.incrementAndGet()
        widen(node)
        if usesRollouts then
          val (rolloutValue, moves) = LightPlayout.playout(node.state, config.maxRolloutDepth, rng)
          value = rolloutValue
          rolloutMoves = moves
        else value = v
        descending = false
      else
        widen(node)
        val child = selectChild(node, rng)
        if child == null then
          // No legal moves at all; treat the position as terminal.
          value = terminalValue(node.state)
          descending = false
        else
          node = child
          path += node
          applyVirtualLoss(node)

    // Backpropagate.  `value` is from the leaf's player-to-move perspective,
    // while a node stores values from the perspective of the player who moved
    // into it, so the sign flips at every level.
    var v = -value.toDouble
    var i = path.length - 1
    while i >= 0 do
      path(i).totalValue.add(v + config.virtualLoss)
      v = -v
      i -= 1

    if config.raveEnabled && rolloutMoves != null then
      updateRave(path, path.length - 1, value, rolloutMoves)

  private def applyVirtualLoss(node: MCTSNode): Unit =
    node.visits.incrementAndGet()
    if config.virtualLoss != 0.0 then node.totalValue.add(-config.virtualLoss)

  /**
   * PUCT selection over the children that exist so far:
   *
   * {{{
   *   score = Q_combined + c_puct * P * sqrt(N_parent) / (1 + N_child)
   * }}}
   *
   * where `Q_combined` blends the tree value with the RAVE value.
   */
  private def selectChild(node: MCTSNode, rng: java.util.Random): MCTSNode | Null =
    val children = node.children
    if children.isEmpty then null
    else
      val sqrtParent = math.sqrt(math.max(1, node.n).toDouble)
      var best: MCTSNode = null
      var bestScore = Double.NegativeInfinity
      var i = 0
      while i < children.length do
        val child = children(i)
        val q = combinedValue(child)
        val exploration = config.c_puct * child.prior * sqrtParent / (1.0 + child.n)
        // A whisper of noise breaks exact ties so that parallel threads do not
        // all commit to the same branch in the same instant.
        val score = q + exploration + rng.nextDouble() * 1e-6
        if score > bestScore then
          bestScore = score
          best = child
        i += 1
      best

  /** Blends the MCTS and RAVE values using the beta schedule from the design doc. */
  private def combinedValue(child: MCTSNode): Double =
    if !config.raveEnabled then child.q
    else
      val n = child.n.toDouble
      val k = config.raveEquivalence
      val beta = math.sqrt(k / (3.0 * n + k))
      (1.0 - beta) * child.q + beta * child.raveQ

  /**
   * Progressive widening (design doc section 2.4).
   *
   * A node may only have `max(1, C * N^alpha)` children.  Because candidates
   * are sorted by prior, the children that do exist are always the ones the
   * policy network likes best.
   */
  private def widen(node: MCTSNode): Unit =
    val candidates = node.candidates
    if candidates != null && candidates.nonEmpty then
      val target =
        if !config.progressiveWidening then candidates.length
        else
          val widened = config.wideningC * math.pow(math.max(1, node.n).toDouble, config.wideningAlpha)
          math.max(1, math.min(candidates.length, math.ceil(widened).toInt))
      val priors = node.candidatePriors
      var i = 0
      while node.childCount < target && i < candidates.length do
        val moveIndex = candidates(i)
        if !node.childMap.containsKey(moveIndex) then
          val p = if moveIndex == node.state.area then Point.Pass else Point(moveIndex)
          node.state.place(p, node.state.toMove) match
            case Right(next) =>
              val child = new MCTSNode(p, priors(i), node, next)
              if node.childMap.putIfAbsent(moveIndex, child) == null then
                nodesCreated.incrementAndGet()
            case Left(_) => () // not among the candidates, so not expected
        i += 1

  /**
   * RAVE / AMAF credit (design doc section 2.3).
   *
   * Classic AMAF: if a move was played *anywhere* in the playout, credit it as
   * if it had been played first at every ancestor where the same player was to
   * move.  Two details make this correct:
   *
   *   - colours alternate, so the rollout move at position `j` was played by
   *     `leafPlayer` when `j` is even and by the opponent when `j` is odd;
   *   - an ancestor `d` plies above the leaf has the same player to move as the
   *     leaf when `d` is even, and the value must be negated when it is not.
   *
   * Only children that already exist are credited, and only the first
   * [[SearchConfig.raveHorizon]] moves are considered.  Full AMAF is
   * `O(path x playout)`, which dominates the search cost long before it helps;
   * the horizon keeps it well under the cost of the rollout itself.
   */
  private def updateRave(
      path: mutable.ArrayBuffer[MCTSNode],
      leafIndex: Int,
      leafValue: Float,
      rolloutMoves: Array[Int]
  ): Unit =
    val horizon = math.min(rolloutMoves.length, config.raveHorizon)
    // Start at the leaf's parent: the leaf itself has no children to compare.
    var up = leafIndex - 1
    while up >= 0 do
      val ancestor = path(up)
      val distance = leafIndex - up
      val samePlayer = distance % 2 == 0
      val valueForAncestor = if samePlayer then leafValue.toDouble else -leafValue.toDouble
      var j = if samePlayer then 0 else 1
      while j < horizon do
        val child = ancestor.childFor(rolloutMoves(j))
        if child != null then
          child.raveVisits.incrementAndGet()
          child.raveTotalValue.add(valueForAncestor)
        j += 2
      up -= 1

  /** The current best move, i.e. the most visited child of the root. */
  def bestMove: Point =
    val children = root.children
    if children.isEmpty then Point.Pass
    else children.maxBy(_.n).move

  /** Visit counts over the root's children. */
  def visitCounts: Map[Point, Int] =
    root.children.map(c => c.move -> c.n).toMap

  /**
   * The normalised visit distribution, which is the improved policy that
   * self-play records as a training target.  Falls back to the network priors
   * when no playout has completed.
   */
  def visitDistribution: Map[Point, Float] =
    val children = root.children
    val total = children.map(_.n).sum
    if total > 0 then children.map(c => c.move -> (c.n.toFloat / total)).toMap
    else if root.isEvaluated then
      val candidates = root.candidates
      val priors = root.candidatePriors
      val sum = priors.sum
      candidates.indices.map { i =>
        val p = if candidates(i) == root.state.area then Point.Pass else Point(candidates(i))
        p -> (if sum > 0 then priors(i) / sum else 0f)
      }.toMap
    else Map.empty

  /** Samples a move from the visit distribution at the given temperature. */
  def sampleMove(temperature: Double, rng: java.util.Random): Point =
    val children = root.children
    if children.isEmpty then Point.Pass
    else if temperature <= 1e-3 then children.maxBy(_.n).move
    else
      val weights = children.map(c => math.pow(c.n.toDouble, 1.0 / temperature))
      val sum = weights.sum
      if sum <= 0 then children.maxBy(_.n).move
      else
        var pick = rng.nextDouble() * sum
        var i = 0
        while i < children.length && pick > weights(i) do
          pick -= weights(i)
          i += 1
        children(math.min(i, children.length - 1)).move

  /** Win rate from the perspective of the player to move at the root. */
  def winRate: Double =
    val children = root.children
    val total = children.map(_.n).sum
    if total == 0 then 0.0
    else
      // A child's Q is from the mover's perspective, which is the root player,
      // so the visit-weighted average is the root player's expected score.
      children.map(c => c.q * c.n).sum / total

  /** The most-visited line, up to a sensible depth. */
  def principalVariation(limit: Int = 20): Vector[Point] =
    val out = Vector.newBuilder[Point]
    var node: MCTSNode = root
    var depth = 0
    var running = true
    while running && depth < limit do
      val children = node.children
      val best = children.filter(_.n > 0).maxByOption(_.n)
      best match
        case Some(child) =>
          out += child.move
          node = child
          depth += 1
        case None => running = false
    out.result()

  def result(elapsed: FiniteDuration): SearchResult =
    SearchResult(
      move = bestMove,
      policy = visitDistribution,
      visits = visitCounts,
      playouts = completedPlayouts.get(),
      elapsed = elapsed,
      winRate = winRate,
      principalVariation = principalVariation(),
      bestPrior = root.children.find(_.move == bestMove).map(_.prior).getOrElse(0f),
      nodesCreated = nodesCreated.get()
    )

object Search:
  /**
   * Searches `state` and returns the best move plus the improved policy.
   *
   * This is the signature given in the design document.
   */
  def bestMove(
      state: BoardState,
      network: PolicyValueNetwork,
      playouts: Int = 1600,
      threads: Int = Runtime.getRuntime.availableProcessors(),
      timeLimit: Option[FiniteDuration] = None
  ): (Point, Map[Point, Float]) =
    val result = search(
      state,
      network,
      SearchConfig(playouts = playouts, threads = threads, timeLimit = timeLimit)
    )
    (result.move, result.policy)

  /** Full search, returning statistics as well as the move. */
  def search(
      state: BoardState,
      network: PolicyValueNetwork,
      config: SearchConfig = SearchConfig()
  ): SearchResult =
    val session = newSession(state, network, config)
    val started = System.nanoTime()
    session.runPlayouts(config.playouts, config.timeLimit.map(t => started + t.toNanos))
    session.result(FiniteDuration(System.nanoTime() - started, TimeUnit.NANOSECONDS))

  /** Creates a search tree without running any playouts, for pondering. */
  def newSession(
      state: BoardState,
      network: PolicyValueNetwork,
      config: SearchConfig = SearchConfig()
  ): SearchSession =
    new SearchSession(network, config, new MCTSNode(Point.Pass, 1f, null, state))

  /**
   * The value of a finished position, from the perspective of `state.toMove`.
   *
   * Decisive results sit near +1/-1; close games give a graded signal via
   * `tanh`, which gives search something to work with beyond a bare win/loss.
   */
  def terminalValue(state: BoardState): Float = terminalValueFrom(state, state.toMove)

  /** The value of a finished position, from the perspective of `player`. */
  def terminalValueFrom(state: BoardState, player: Color): Float =
    require(state.isTerminal, "terminalValueFrom needs a finished position")
    state.resigned match
      case Some(loser) => if player == loser then -1f else 1f
      case None =>
        val margin = state.score.fromPerspective(player)
        math.tanh(margin / (0.08 * state.area)).toFloat

  /** A symmetric Dirichlet sample, used for root exploration in self-play. */
  private[go] def dirichlet(count: Int, alpha: Double, rng: java.util.Random): Array[Double] =
    val out = new Array[Double](count)
    var sum = 0.0
    var i = 0
    while i < count do
      val g = gamma(alpha, rng)
      out(i) = if g <= 0.0 then 1e-12 else g
      sum += out(i)
      i += 1
    if sum <= 0.0 then java.util.Arrays.fill(out, 1.0 / count)
    else
      i = 0
      while i < count do
        out(i) = out(i) / sum
        i += 1
    out

  /**
   * Standard Marsaglia-Tsang gamma sampler, needed because the Dirichlet
   * distribution is a normalised vector of Gamma(alpha, 1) draws and the JDK
   * has no gamma generator.
   */
  private def gamma(alpha: Double, rng: java.util.Random): Double =
    if alpha < 1.0 then
      // Boost up to alpha + 1 and use the identity for small shape parameters.
      val u = rng.nextDouble()
      gamma(alpha + 1.0, rng) * math.pow(u, 1.0 / alpha)
    else
      val d = alpha - 1.0 / 3.0
      val c = 1.0 / math.sqrt(9.0 * d)
      var result = -1.0
      while result < 0.0 do
        var x = 0.0
        var v = 0.0
        var accepted = false
        while !accepted do
          x = gaussian(rng)
          v = 1.0 + c * x
          if v > 0.0 then
            v = v * v * v
            val u = rng.nextDouble()
            if u < 1.0 - 0.0331 * x * x * x * x then accepted = true
            else if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)) then accepted = true
        result = d * v
      result

  /** Standard normal via the polar Box-Muller transform. */
  private def gaussian(rng: java.util.Random): Double =
    var u1 = rng.nextDouble()
    while u1 <= 1e-12 do u1 = rng.nextDouble()
    val u2 = rng.nextDouble()
    math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.Pi * u2)

// ----------------------------------------------------------------------------
// Design doc section 2.6 -- alpha-beta tactical supplement.
//
// Deliberately not implemented.  A trustworthy Go alpha-beta search needs its
// own move generation for capturing races, threat extensions and a static
// exchange evaluator; a half-hearted version would be worse than none, and
// nothing else in the engine depends on it.  The natural home for it is a new
// `Tactics.scala` that `SearchConfig` can opt into for semeai/tsumego
// subproblems.  This comment is the marker for that work.
// ----------------------------------------------------------------------------

/**
 * Time management: sudden death, byo-yomi and per-move budgets.
 *
 * GTP reports a main time plus, optionally, a byo-yomi period repeated a
 * number of times.  [[Clock]] tracks how much of that is left and decides how
 * long the next move may take.
 */
final case class ByoYomi(periodTime: FiniteDuration, periods: Int):
  require(periods > 0, "byo-yomi needs at least one period")

final case class TimeControl(
    mainTime: FiniteDuration = FiniteDuration(5, "min"),
    byoYomi: Option[ByoYomi] = None,
    /** Reserved for transmission latency and process overhead. */
    moveOverhead: FiniteDuration = FiniteDuration(1, "s")
):
  /** Sudden death only. */
  def isSuddenDeath: Boolean = byoYomi.isEmpty

object TimeControl:
  def suddenDeath(main: FiniteDuration): TimeControl = TimeControl(main, None)
  def byoYomi(main: FiniteDuration, period: FiniteDuration, periods: Int): TimeControl =
    TimeControl(main, Some(ByoYomi(period, periods)))

/**
 * A mutable clock for one player.
 *
 * `timeForMove` divides the remaining main time over an estimate of how many
 * moves are left, and switches to a fraction of the byo-yomi period once the
 * main time is exhausted.
 */
final class Clock(val control: TimeControl):
  private var mainRemaining: Long = control.mainTime.toNanos
  private var periodRemaining: Long = control.byoYomi.map(_.periodTime.toNanos).getOrElse(0L)
  private var periodsLeft: Int = control.byoYomi.map(_.periods).getOrElse(0)
  private var spent: Long = 0L

  def inByoYomi: Boolean = mainRemaining <= 0 && periodsLeft > 0

  def mainTimeLeft: FiniteDuration = FiniteDuration(math.max(0L, mainRemaining), TimeUnit.NANOSECONDS)

  def periodsRemaining: Int = periodsLeft

  /**
   * How long the next move may take.  `movesToGo` comes from GTP's
   * `time_left ... <moves>` and improves the estimate when available.
   */
  def timeForMove(movesToGo: Option[Int] = None): FiniteDuration =
    val overhead = control.moveOverhead.toNanos
    if inByoYomi then
      val usable = math.max(0L, periodRemaining - overhead)
      // Leave some of the period in hand rather than spending it all at once.
      FiniteDuration(math.max(10_000_000L, usable / 3), TimeUnit.NANOSECONDS)
    else
      val estimatedMoves = math.max(movesToGo.getOrElse(20), 1)
      val usable = math.max(0L, mainRemaining - overhead)
      // Spend at most a quarter of what is left, so one move cannot eat the game.
      val share = math.min(usable / estimatedMoves, usable / 4)
      FiniteDuration(math.max(10_000_000L, share), TimeUnit.NANOSECONDS)

  /** Records the time a move actually took, consuming byo-yomi periods if needed. */
  def recordMove(elapsed: FiniteDuration): Unit =
    val used = elapsed.toNanos
    spent += used
    if inByoYomi then
      periodRemaining -= used
      if periodRemaining <= 0 then
        periodsLeft -= 1
        periodRemaining = control.byoYomi.map(_.periodTime.toNanos).getOrElse(0L)
    else
      mainRemaining -= used
      if mainRemaining < 0 then
        // Main time is gone; carry the excess into the first byo-yomi period.
        val excess = -mainRemaining
        mainRemaining = 0
        if periodsLeft > 0 then
          periodRemaining -= excess
          if periodRemaining <= 0 then
            periodsLeft -= 1
            periodRemaining = control.byoYomi.map(_.periodTime.toNanos).getOrElse(0L)

  def totalSpent: FiniteDuration = FiniteDuration(spent, TimeUnit.NANOSECONDS)

  override def toString: String =
    val byo = if periodsLeft > 0 then s", byo-yomi x$periodsLeft" else ", sudden death"
    s"${mainTimeLeft.toSeconds}s left$byo"
