//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import java.nio.FloatBuffer

// ============================================================================
// Evaluate.scala -- position evaluation.
//
// Three interchangeable implementations of the same trait:
//
//   RandomNetwork     uniform policy, hash-derived value.  No knowledge at all.
//   HeuristicNetwork  hand-written policy and influence value.  This is the
//                     default, and it makes the engine genuinely playable with
//                     no model file and no dependencies.
//   OnnxNetwork       a trained policy/value network on the Apple GPU.
//
// IMPORTANT - dependency policy
//   Everything except OnnxNetwork compiles and runs with zero dependencies.
//   OnnxNetwork talks to ONNX Runtime through java.lang.reflect, so the core
//   project never needs the onnxruntime jar on its classpath.  Add it only for
//   the runs that actually need a model:
//
//       make onnx-run MODEL=models/gen0.onnx
//       # or: scala-cli run . --dep com.microsoft.onnxruntime:onnxruntime:1.20.0 \
//       #                    --main-class go.onnxCheck -- models/gen0.onnx
//
//   This keeps `make check` fast and offline while still giving the trained
//   network a real, GPU-accelerated path.  See README.md.
//
// VALUE CONVENTION -- read this once, it matters everywhere
//   `evaluate(state)` returns the value from the perspective of the player who
//   is TO MOVE in `state`, in [-1, +1], where +1 means "the player to move is
//   winning".  Search.scala negates this as it backs values up the tree.
// ============================================================================

/**
 * A policy and value network.
 *
 * The policy is a probability distribution over `size * size` intersections
 * plus a pass move, whose index is `state.area`.  The value is a scalar in
 * `[-1, +1]` from the point of view of `state.toMove`.
 *
 * Implementations must be safe to call from many threads at once: parallel
 * MCTS evaluates leaves concurrently.
 */
trait PolicyValueNetwork:
  /** Short identifier, shown in the CLI and in GTP `name` replies. */
  def name: String

  /** `(policy, value)` for one position.  Policy length is `state.area + 1`. */
  def evaluate(state: BoardState): (Array[Float], Float)

  /**
   * Evaluates several positions.  The default just loops; [[OnnxNetwork]]
   * overrides it to use a single batched inference, which is much faster on
   * the GPU and is what makes parallel MCTS worthwhile.
   */
  def evaluateBatch(states: Seq[BoardState]): Seq[(Array[Float], Float)] =
    states.map(evaluate)

  /**
   * False for networks whose value output carries no information, so that
   * search knows to fall back on random rollouts instead of trusting it.
   */
  def valueIsMeaningful: Boolean = true

  /** Index of `p` in the policy vector; pass occupies the final slot. */
  def policyIndex(p: Point, state: BoardState): Int =
    if p.isPass then state.area else p.index

  /** Rejects a policy vector that does not match the board. */
  protected def requireShape(policy: Array[Float], state: BoardState): Unit =
    require(
      policy.length == state.area + 1,
      s"policy length ${policy.length} does not match board area ${state.area} + 1 pass"
    )

/**
 * The 15 input feature planes described in section 3.2 of the design document.
 *
 * Encoding a position this way, rather than handing the raw grid to a network,
 * is what lets a convolutional network reason about liberties and recent
 * history.  Planes are returned as `Array[Array[Float]]` indexed
 * `[plane][point]`, which is exactly the shape `TrainingExample` stores and
 * that `OnnxNetwork` flattens into a `(1, 15, size, size)` tensor.
 *
 * {{{
 *   plane  0   current player's stones
 *   plane  1   opponent's stones
 *   plane  2   liberties of the current player's groups, in [0, 1]
 *   plane  3   liberties of the opponent's groups, in [0, 1]
 *   plane  4   the ko point
 *   planes 5-12  the last four moves; a stone plane and a whole-board
 *                "this move was Black" plane for each
 *   plane 13   side to move (all ones for Black, all zeros for White)
 *   plane 14   tactical hotspots: the single liberty of any group in atari
 * }}}
 */
object FeaturePlanes:
  /** Number of planes produced by [[encode]]. */
  val PlaneCount = 15

  /** Number of recent moves recorded in the history planes. */
  val HistoryMoves = 4

  def encode(state: BoardState, firstColor: Color = Color.Black): Array[Array[Float]] =
    val n = state.area
    val planes = Array.fill(PlaneCount)(new Array[Float](n))
    val me = state.toMove
    val opp = me.opposite
    val cells = state.cells

    // Planes 0-1: the two stone colours.
    var i = 0
    while i < n do
      cells(i) match
        case 1 => if me == Color.Black then planes(0)(i) = 1f else planes(1)(i) = 1f
        case 2 => if me == Color.White then planes(0)(i) = 1f else planes(1)(i) = 1f
        case _ => ()
      i += 1

    // Planes 2-3: liberty counts, bucketed into [0, 1].  Groups are walked once
    // with a visited mask so that a large group is not flood filled 361 times.
    val visited = new Array[Boolean](n)
    i = 0
    while i < n do
      if cells(i) != 0 && !visited(i) then
        val color = Color.fromByte(cells(i))
        val (stones, libs) = state.flood(i)
        val value = math.min(libs.length, 4).toFloat / 4f
        val plane = if color == me then 2 else 3
        var k = 0
        while k < stones.length do
          planes(plane)(stones(k)) = value
          visited(stones(k)) = true
          k += 1
      i += 1

    // Plane 4: the ko point, if any.
    state.koPoint.foreach(p => planes(4)(p.index) = 1f)

    // Planes 5-12: the last few moves.  Move index j was played by firstColor
    // when j is even, which is the normal alternation.
    val history = state.moveHistory
    var slot = 0
    while slot < HistoryMoves do
      val moveIndex = history.length - 1 - slot
      if moveIndex >= 0 then
        val p = history(moveIndex)
        val stonePlane = 5 + slot * 2
        val colorPlane = stonePlane + 1
        if !p.isPass then planes(stonePlane)(p.index) = 1f
        val mover = if moveIndex % 2 == 0 then firstColor else firstColor.opposite
        if mover == Color.Black then java.util.Arrays.fill(planes(colorPlane), 1f)
      slot += 1

    // Plane 13: whose turn it is.
    if me == Color.Black then java.util.Arrays.fill(planes(13), 1f)

    // Plane 14: mark the last liberty of every group in atari.  A cheap
    // tactical signal that helps with ladders and capturing races.
    val marked = new Array[Boolean](n)
    i = 0
    while i < n do
      if cells(i) != 0 && !marked(i) then
        val (stones, libs) = state.flood(i)
        var k = 0
        while k < stones.length do
          marked(stones(k)) = true
          k += 1
        if libs.length == 1 then planes(14)(libs(0)) = 1f
      i += 1

    planes

  /** Flattens `[plane][point]` into one `plane * area` long array. */
  def flatten(planes: Array[Array[Float]]): Array[Float] =
    val n = planes.headOption.map(_.length).getOrElse(0)
    val out = new Array[Float](planes.length * n)
    var p = 0
    while p < planes.length do
      System.arraycopy(planes(p), 0, out, p * n, n)
      p += 1
    out

  /**
   * Builds a `(batch, planes, size, size)` tensor as a flat float buffer, the
   * layout ONNX Runtime expects.
   */
  def toNchw(states: Seq[BoardState]): Array[Float] =
    if states.isEmpty then Array.emptyFloatArray
    else
      val size = states.head.size
      val perSample = PlaneCount * size * size
      val out = new Array[Float](states.length * perSample)
      var b = 0
      while b < states.length do
        System.arraycopy(flatten(encode(states(b))), 0, out, b * perSample, perSample)
        b += 1
      out

/**
 * Uniform policy and a hash-derived value.
 *
 * This is the starting point described in the design document, useful for
 * exercising the search before any network exists.  The "random" value is
 * derived from the Zobrist hash rather than a `Random` instance, which makes
 * it deterministic, reproducible, and safe to call from many threads.
 */
final class RandomNetwork extends PolicyValueNetwork:
  val name = "random"

  override def valueIsMeaningful: Boolean = false

  def evaluate(state: BoardState): (Array[Float], Float) =
    val n = state.area
    val policy = new Array[Float](n + 1)
    val uniform = 1f / (n + 1)
    java.util.Arrays.fill(policy, uniform)
    (policy, RandomNetwork.hashValue(state.zobristHash))

object RandomNetwork:
  /** Maps a 64-bit hash into `[-1, +1)`, deterministically. */
  private[go] def hashValue(hash: Long): Float =
    val mixed = hash * 0x9E3779B97F4A7C15L
    val unit = ((mixed >>> 11).toDouble) / (1L << 53).toDouble
    (unit * 2.0 - 1.0).toFloat

/**
 * A hand-written policy and value function.
 *
 * This is the default network, and the reason the engine is playable on a
 * fresh checkout with no model file.  It is not strong, but it understands the
 * things that keep a beginner engine from embarrassing itself:
 *
 *   - take captures, and rescue friendly groups in atari,
 *   - stay near existing stones,
 *   - never fill your own eye (which would destroy your own group), and
 *   - avoid self-atari and the first line early on.
 *
 * The value estimate is a cheap influence metric: every point is scored by how
 * much closer it is to Black stones than to White stones, plus the stone
 * counts, squashed through `tanh`.  That gives MCTS a usable gradient long
 * before a neural network is available.
 */
final class HeuristicNetwork extends PolicyValueNetwork:
  import HeuristicEval.*

  val name = "heuristic"

  def evaluate(state: BoardState): (Array[Float], Float) =
    val n = state.area
    val policy = new Array[Float](n + 1)
    val me = state.toMove
    val legal = state.legalMoves(me)

    if legal.isEmpty then
      policy(n) = 1f
      (policy, influenceValue(state))
    else
      val weights = new Array[Double](legal.length)
      var best = Double.NegativeInfinity
      var i = 0
      while i < legal.length do
        val w = moveWeight(state, legal(i))
        weights(i) = w
        if w > best then best = w
        i += 1

      // Softmax over the legal moves, shifted by the maximum for stability.
      var sum = 0.0
      i = 0
      while i < legal.length do
        val e = math.exp((weights(i) - best) / Temperature)
        weights(i) = e
        sum += e
        i += 1

      // A small but non-zero chance of passing, so search can choose to stop.
      // The pass probability is reserved out of the total rather than added on
      // top, so that the policy remains a genuine distribution summing to 1.
      val moveShare = (1.0 - PassProbability) / sum
      i = 0
      while i < legal.length do
        policy(legal(i).index) = (weights(i) * moveShare).toFloat
        i += 1
      policy(n) = PassProbability.toFloat
      (policy, influenceValue(state))

  private val Temperature = 0.6
  private val PassProbability = 0.01

  /**
   * A relative score for one legal move.  Larger is better; the absolute scale
   * is irrelevant because the values are passed through a softmax.
   */
  private def moveWeight(state: BoardState, p: Point): Double =
    val me = state.toMove
    val size = state.size
    val nbrs = state.neighborsOf(p)
    val meByte = me.ordinal.toByte
    var w = 1.0

    // Captures are almost always worth taking.
    state.computeMove(p.index, me) match
      case Right(outcome) => w += 3.0 * outcome.captured.length
      case Left(_)        => return 0.0

    // Stay near the action.
    var friendlyNeighbours = 0
    var enemyNeighbours = 0
    var friendlyInAtari = false
    for q <- nbrs do
      state.stoneAt(q) match
        case c if c == me =>
          friendlyNeighbours += 1
        case c if c.isStone =>
          enemyNeighbours += 1
          if state.libertiesAt(q) == 1 then w += 2.5 // capture or atari
        case _ => ()
    if friendlyNeighbours > 0 && nbrs.forall(q => state.stoneAt(q) == me) then
      // Every neighbour is ours: this is an eye, and filling it can kill the
      // very group it belongs to.
      w -= 8.0

    // Rescuing a friendly group that is in atari is usually urgent.
    for q <- nbrs do
      if state.stoneAt(q) == me && state.libertiesAt(q) == 1 then friendlyInAtari = true
    if friendlyInAtari then w += 2.0

    w += 0.6 * friendlyNeighbours + 0.35 * enemyNeighbours

    // Avoid self-atari: a move into a one-liberty group that captures nothing.
    val (_, ownLibs) = state.flood(p.index, p.index, meByte)
    val captures = state.computeMove(p.index, me).toOption.map(_.captured.length).getOrElse(0)
    if ownLibs.length == 1 && captures == 0 then w -= 2.5

    // Opening theory, crudely: the first line is bad, the third and fourth are
    // good, and star points are worth taking early.
    val x = p.x(size)
    val y = p.y(size)
    val edgeDistance = math.min(math.min(x, size - 1 - x), math.min(y, size - 1 - y))
    edgeDistance match
      case 0 => w -= 0.9
      case 1 => w -= 0.35
      case 2 => w += 0.1
      case _ => w += 0.25

    if state.moveCount < 8 && state.starPoints.contains(p) then w += 0.8

    w

/**
 * The influence-based value function shared by the heuristic network and the
 * evaluation bar in the CLI.
 */
object HeuristicEval:
  /** Largest influence distance considered, in Manhattan steps. */
  private val MaxDistance = 4

  /**
   * Estimates the position from the point of view of `state.toMove` in
   * `[-1, +1]`, by comparing how strongly each empty point leans to each side.
   */
  def influenceValue(state: BoardState): Float =
    val balance = balanceFor(state, Color.Black)
    val fromBlack = math.tanh(balance / (0.6 * state.area))
    val signed = if state.toMove == Color.Black then fromBlack else -fromBlack
    signed.toFloat

  /**
   * A per-point influence estimate, for display.
   *
   * Positive means Black dominates the point, negative means White does, and
   * values near zero are neutral.  Influence decays with distance from the
   * nearest stone of each colour, which is a crude but useful picture of who
   * owns what -- it is what the CLI's evaluation display is drawn from.
   *
   * This is a heuristic for humans, not a scored quantity: use
   * [[balanceFor]] when a single number is wanted.
   */
  def influenceAt(state: BoardState, point: Point): Double =
    if !point.onBoard(state.size) then 0.0
    else
      val black = influenceDecay(distanceField(state, Color.Black)(point.index))
      val white = influenceDecay(distanceField(state, Color.White)(point.index))
      black - white

  /** Influence falls off with graph distance; 4 is the far edge of influence. */
  private def influenceDecay(distance: Int): Double =
    if distance > MaxDistance then 0.0 else math.pow(0.55, distance.toDouble)

  /** Positive means Black is ahead. */
  def balanceFor(state: BoardState, _perspective: Color): Double =
    val size = state.size
    val n = state.area
    val cells = state.cells
    val db = distanceField(state, Color.Black)
    val dw = distanceField(state, Color.White)

    var sum = 0.0
    var stones = 0
    var i = 0
    while i < n do
      cells(i) match
        case 1 => stones += 1
        case 2 => stones -= 1
        case _ =>
          // Whoever is closer owns the point; the clamp keeps one nearby stone
          // from dominating the whole board.
          val diff = dw(i) - db(i)
          sum += math.max(-MaxDistance, math.min(MaxDistance, diff))
        i += 1
    // Stones themselves are worth a full point each, on the same scale.
    sum + MaxDistance * stones

  /**
   * Multi-source breadth-first search giving, for every point, the Manhattan
   * distance to the nearest stone of `color`, capped at [[MaxDistance]].
   */
  private def distanceField(state: BoardState, color: Color): Array[Int] =
    val size = state.size
    val n = state.area
    val cells = state.cells
    val target = color.ordinal.toByte
    val dist = Array.fill(n)(MaxDistance + 1)
    val queue = new Array[Int](n)
    var head = 0
    var tail = 0

    var i = 0
    while i < n do
      if cells(i) == target then
        dist(i) = 0
        queue(tail) = i
        tail += 1
      i += 1

    val nbrs = Geometry.neighbors(size)
    while head < tail do
      val cur = queue(head)
      head += 1
      val d = dist(cur)
      if d < MaxDistance then
        val adj = nbrs(cur)
        var k = 0
        while k < adj.length do
          val nb = adj(k)
          if dist(nb) > d + 1 then
            dist(nb) = d + 1
            queue(tail) = nb
            tail += 1
          k += 1
    dist

/**
 * Runs trained ONNX models (policy and value heads) on the Apple GPU.
 *
 * The design document specifies ONNX Runtime with the CoreML execution
 * provider, which maps onto Metal.  This class reaches that API reflectively so
 * that the project compiles and runs without the `onnxruntime` jar; the jar is
 * needed only on runs that actually load a model.  See the header of this file
 * and the `make onnx-*` targets.
 *
 * Expected model signature:
 * {{{
 *   input   "input"  float32[1, 15, size, size]   from FeaturePlanes.toNchw
 *   output 0         float32[1, size*size + 1]    policy (raw logits or probs)
 *   output 1         float32[1, 1]                value in [-1, +1]
 * }}}
 */
final class OnnxNetwork(
    modelPath: String,
    useCoreML: Boolean = true,
    inputName: String = "input",
    policyOutput: Int = 0,
    valueOutput: Int = 1
) extends PolicyValueNetwork:
  private val session = OnnxBridge.createSession(modelPath, useCoreML)

  val name: String = s"onnx:${java.nio.file.Path.of(modelPath).getFileName}${
      if useCoreML then " (coreml)" else " (cpu)"
    }"

  def evaluate(state: BoardState): (Array[Float], Float) =
    evaluateBatch(Seq(state)).head

  override def evaluateBatch(states: Seq[BoardState]): Seq[(Array[Float], Float)] =
    if states.isEmpty then Seq.empty
    else
      val size = states.head.size
      states.foreach { s =>
        require(s.size == size, "all positions in a batch must share a board size")
      }
      val data = FeaturePlanes.toNchw(states)
      val shape = Array(states.length.toLong, FeaturePlanes.PlaneCount.toLong, size.toLong, size.toLong)
      // `run` copies everything into plain Java arrays and closes the native
      // tensors and the result itself, so nothing here needs releasing.
      val outputs = OnnxBridge.run(session, inputName, data, shape)
      val rawPolicy = outputs.lift(policyOutput).getOrElse(Array.empty[Array[Float]])
      val rawValue = outputs.lift(valueOutput).getOrElse(Array.empty[Array[Float]])
      states.indices.map { b =>
        val policy = decodePolicy(rawPolicy.lift(b).getOrElse(Array.empty[Float]), states(b))
        (policy, valueFor(rawValue, b, states.length))
      }

  /**
   * Reads the value for batch element `b`.
   *
   * The documented output shape is `(batch, 1)`, but a model exported with a
   * flattened value head produces `(batch,)`; both are accepted rather than
   * trusting the export.
   */
  private def valueFor(raw: Array[Array[Float]], b: Int, batch: Int): Float =
    if raw.length >= batch && raw(b).nonEmpty then raw(b)(0)
    else if raw.length == 1 then raw(0).lift(b).orElse(raw(0).headOption).getOrElse(0f)
    else 0f

  /** Turns raw network output into a probability distribution. */
  private def decodePolicy(row: Array[Float], state: BoardState): Array[Float] =
    val expected = state.area + 1
    // Always copy: a longer row is truncated to the board's move count and a
    // shorter one is zero-padded, and the normalisation below works in place.
    val trimmed = java.util.Arrays.copyOf(row, expected)

    // Accept both raw logits and an already-normalised distribution.
    val total = trimmed.sum
    val looksNormalised =
      trimmed.forall(v => v >= 0f && v <= 1f) && math.abs(total - 1f) < 1e-3f
    if looksNormalised then trimmed
    else
      var best = Float.NegativeInfinity
      var i = 0
      while i < trimmed.length do
        if trimmed(i) > best then best = trimmed(i)
        i += 1
      var sum = 0.0
      i = 0
      while i < trimmed.length do
        val e = math.exp((trimmed(i) - best).toDouble)
        trimmed(i) = e.toFloat
        sum += e
        i += 1
      i = 0
      while i < trimmed.length do
        trimmed(i) = (trimmed(i) / sum).toFloat
        i += 1
      trimmed

/**
 * The reflective bridge to ONNX Runtime.
 *
 * Every call below is a normal, documented part of the ONNX Runtime Java API;
 * reflection is used only so the dependency stays optional.  Any failure is
 * reported as an `IllegalStateException` naming the missing piece and the
 * command that fixes it, rather than as a `NoClassDefFoundError` deep inside
 * the search.
 */
object OnnxBridge:
  private val EnvironmentClass = "ai.onnxruntime.OrtEnvironment"
  private val SessionOptionsClass = "ai.onnxruntime.OrtSession$SessionOptions"
  private val SessionClass = "ai.onnxruntime.OrtSession"
  private val TensorClass = "ai.onnxruntime.OnnxTensor"
  private val ResultClass = "ai.onnxruntime.OrtSession$Result"

  /** True when the onnxruntime jar is on the classpath. */
  def isAvailable: Boolean =
    try
      Class.forName(EnvironmentClass)
      true
    catch case _: Throwable => false

  /** Explains how to make [[isAvailable]] true. */
  def missingDependencyMessage: String =
    s"""ONNX Runtime is not on the classpath, so no trained model can be loaded.
       |The core engine deliberately has no dependencies, so ONNX support is opt-in:
       |
       |    make onnx-run MODEL=path/to/model.onnx
       |
       |or directly:
       |
       |    scala-cli run . \\
       |      --dep com.microsoft.onnxruntime:onnxruntime:$SupportedOnnxVersion \\
       |      --main-class go.CLI -- --model path/to/model.onnx
       |
       |Everything except the ONNX backend works without it.  See README.md.""".stripMargin

  /** The version the Makefile and docs use by default. */
  val SupportedOnnxVersion = "1.20.0"

  /** A live ONNX Runtime session, boxed so no ONNX type leaks into our API. */
  final class Session(private[OnnxBridge] val underlying: AnyRef, private[OnnxBridge] val env: AnyRef)

  def createSession(modelPath: String, useCoreML: Boolean): Session =
    if !isAvailable then throw new IllegalStateException(missingDependencyMessage)
    if !java.nio.file.Files.exists(java.nio.file.Path.of(modelPath)) then
      throw new IllegalArgumentException(s"ONNX model not found: $modelPath")
    try
      val envClass = Class.forName(EnvironmentClass)
      val env = envClass.getMethod("getEnvironment").invoke(null)
      val optionsClass = Class.forName(SessionOptionsClass)
      val options = optionsClass.getConstructor().newInstance()
      if useCoreML then
        // Falls back silently when CoreML is unavailable (e.g. on an Intel Mac).
        try optionsClass.getMethod("addCoreML").invoke(options)
        catch case _: Throwable => ()
      val session = Class
        .forName(SessionClass)
        .getMethod("createSession", classOf[String], optionsClass)
        .invoke(env, modelPath, options)
      Session(session, env)
    catch
      case e: java.lang.reflect.InvocationTargetException =>
        throw new IllegalStateException(
          s"ONNX Runtime failed to create a session for $modelPath: ${rootMessage(e)}",
          e
        )

  /**
   * Loads the features into a tensor, runs the session and returns the outputs
   * as `Array[Array[Array[Float]]]`, one matrix per model output position.
   *
   * Everything the runtime allocates -- the input tensor, the `OrtSession.Result`
   * and the output tensors inside it -- is released before this returns, which
   * is why the outputs are copied into plain Java arrays first.  Holding on to
   * the result instead is the easy way to leak native memory on every move.
   */
  def run(
      session: Session,
      inputName: String,
      data: Array[Float],
      shape: Array[Long]
  ): Array[Array[Array[Float]]] =
    try
      val envClass = Class.forName(EnvironmentClass)
      val tensorClass = Class.forName(TensorClass)
      val tensor = tensorClass
        .getMethod("createTensor", envClass, classOf[FloatBuffer], classOf[Array[Long]])
        .invoke(null, session.env, FloatBuffer.wrap(data), shape)
      try
        val inputs = java.util.Map.of(inputName, tensor)
        val result = Class
          .forName(SessionClass)
          .getMethod("run", classOf[java.util.Map[?, ?]])
          .invoke(session.underlying, inputs)
        val resultClass = Class.forName(ResultClass)
        try
          // Read every output position we might need, tolerating a model with a
          // single output or an output the runtime refuses to hand over.
          Array(0, 1).map { index =>
            try toFloatMatrix(resultClass.getMethod("get", classOf[Int]).invoke(result, Int.box(index)))
            catch case _: Throwable => Array.empty[Array[Float]]
          }
        finally closeQuietly(result)
      finally closeQuietly(tensor)
    catch
      case e: java.lang.reflect.InvocationTargetException =>
        throw new IllegalStateException(s"ONNX Runtime failed to run the model: ${rootMessage(e)}", e)

  /** Unwraps the value held by an `OrtSession.Result` entry into a float row. */
  def toFloatMatrix(value: AnyRef): Array[Array[Float]] =
    if value == null then throw new IllegalStateException("ONNX model did not produce the expected output")
    val raw = value.getClass.getMethod("getValue").invoke(value)
    raw match
      case a: Array[Array[Float]] => a
      case a: Array[Float]        => Array(a)
      case other =>
        throw new IllegalStateException(
          s"unsupported ONNX output type: ${if other == null then "null" else other.getClass.getName}"
        )

  private def closeQuietly(value: AnyRef): Unit =
    if value != null then
      try value.getClass.getMethod("close").invoke(value)
      catch case _: Throwable => ()

  private def rootMessage(e: Throwable): String =
    val cause = if e.getCause != null then e.getCause else e
    s"${cause.getClass.getSimpleName}: ${cause.getMessage}"

// ============================================================================
// Optional ONNX diagnostic
// ============================================================================

/**
 * `make onnx-check` -- reports whether the optional ONNX Runtime is available.
 *
 * With a model it also evaluates the opening position once, which is the
 * fastest way to confirm that the input and output names of an exported model
 * match what [[OnnxNetwork]] expects before starting a game.
 */
@main def onnxCheck(args: String*): Unit =
  val parsed = Args.parse(args.toArray)
  println(s"ONNX Runtime on the classpath: ${OnnxBridge.isAvailable}")

  if !OnnxBridge.isAvailable then
    println()
    println(OnnxBridge.missingDependencyMessage)
  else
    parsed.get("model") match
      case None =>
        println("no --model given; pass MODEL=path/to/model.onnx to evaluate a position")
      case Some(path) if !java.nio.file.Files.exists(java.nio.file.Paths.get(path)) =>
        println(s"no such model file: $path")
      case Some(path) =>
        println(s"loading $path")
        try
          val network = OnnxNetwork(path, useCoreML = !parsed.bool("no-coreml", false))
          println(s"network name: ${network.name}")
          val state = BoardState.initial(parsed.int("size", 9), parsed.double("komi", 6.5))
          val (policy, value) = network.evaluate(state)
          println(
            f"opening position: value ${value}%.4f from ${state.toMove}'s point of view, " +
              f"policy of ${policy.length} moves summing to ${policy.sum}%.4f"
          )
          println("top moves:")
          val ranked = policy.indices
            .map { i =>
              val point: Point = if i == state.area then Point.Pass else Point(i)
              (point, policy(i))
            }
            .sortBy(-_._2)
            .take(5)
          for (p, prob) <- ranked do
            println(f"  ${p.toGtp(state.size)}%-5s ${prob * 100}%5.1f%%")
        catch
          case e: Throwable =>
            println(s"failed to evaluate with this model: ${e.getMessage}")
            System.exit(1)
