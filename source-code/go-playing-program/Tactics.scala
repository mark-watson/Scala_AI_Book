//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import scala.collection.mutable

// ============================================================================
// Tactics.scala -- alpha-beta tactical supplement (DESIGN.md section 2.6).
//
// MCTS spreads its visits thin: a group with three liberties in a closed
// region can take thousands of playouts to capture by search alone, while a
// three-ply liberty-filling line is obvious to a direct reader.  This file is
// that reader: a depth-bounded alpha-beta over the local region around one
// victim group, answering "can the attacker force this capture".
//
// Scope, read before trusting an answer:
//   * LOCAL analysis.  Attacker moves are confined to the victim's liberties
//     (plus moves that capture an adjacent stone); defender moves to the
//     neighbourhood.  A group that can be saved from OUTSIDE the region -- a
//     connecting move two points away, a ko threat elsewhere -- is outside
//     what this sees.  Positions with an active ko are therefore declined
//     outright: ko fights live outside any local region by definition.
//   * SOUND in one direction only.  `Some(move)` means every defender reply in
//     the region still loses the group: a proven line, never a guess.  `None`
//     means "not proven", not "alive" -- the group may still be dead by a
//     longer or wider line than the bounds allow.
//   * BOUNDED.  Victims with more than a handful of liberties, or regions
//     larger than a small neighbourhood, are declined without searching, and
//     every call has a node budget.  This keeps the worst case in the
//     milliseconds, which matters because scoring calls this on hot paths.
//
// The MCTS-facing entry point is [[findForcedCapture]]; [[SearchConfig.tactics]]
// opts the search into boosting a proven capture's prior.  [[LifeDeath]] uses
// [[canForceCapture]] to decide dead groups for scoring and GTP status lists.
// ============================================================================

/** A proven forced capture: play `firstMove`, and the victim falls by force. */
final case class TacticalLine(
    /** The move to play now; always legal in the analysed position. */
    firstMove: Point,
    /** Stones in the victim group at analysis time. */
    victimStones: Int,
    /** Search depth that proved the line, in plies. */
    depth: Int
)

object Tactics:
  /** Largest victim, in liberties, this will attempt to read. */
  val MaxVictimLiberties = 6

  /** Largest local region, in points, this will search. */
  val MaxRegionPoints = 64

  /** Default ceiling on searched nodes per call. */
  val DefaultNodeBudget = 100000

  /**
   * The first proven capture for `attacker`, or `None` when nothing is proven.
   *
   * Targets are tried smallest-first: an atari group is checked before a
   * three-liberty group, so the common case returns after one cheap proof.
   */
  def findForcedCapture(
      state: BoardState,
      attacker: Color = null,
      maxDepth: Int = 8,
      nodeBudget: Int = DefaultNodeBudget
  ): Option[TacticalLine] =
    val side = if attacker == null then state.toMove else attacker
    if state.koPoint.isDefined then None
    else
      val victims = state.groups.values.toVector.distinct
        .filter(_.color == side.opposite)
        .sortBy(g => (g.liberties.size, -g.size))
      val budget = new NodeBudget(nodeBudget)
      victims.collectFirst(Function.unlift { victim =>
        if victim.liberties.size > MaxVictimLiberties then None
        else
          forcedCaptureMove(state, victim.stones.head, side, maxDepth, budget)
            .map(move => TacticalLine(move, victim.size, maxDepth))
      })

  /** True when `attacker` can force the capture of the group at `target`. */
  def canForceCapture(
      state: BoardState,
      target: Point,
      attacker: Color,
      maxDepth: Int = 8,
      nodeBudget: Int = DefaultNodeBudget
  ): Boolean =
    forcedCaptureMove(state, target, attacker, maxDepth, new NodeBudget(nodeBudget)).isDefined

  /**
   * The winning first move against the group at `target`, or `None`.
   *
   * `None` covers three cases that callers must not distinguish: the group is
   * safe, the position is out of scope (active ko, too open), or the proof did
   * not fit in the depth and node budgets.
   */
  def forcedCaptureMove(
      state: BoardState,
      target: Point,
      attacker: Color,
      maxDepth: Int = 8,
      budget: NodeBudget = new NodeBudget(DefaultNodeBudget)
  ): Option[Point] =
    if state.koPoint.isDefined then return None
    val victim = state.groupAt(target)
    if victim.isEmpty || victim.get.color == attacker then return None
    val group = victim.get
    if group.liberties.size > MaxVictimLiberties then return None
    val region = localRegion(state, group)
    if region.size > MaxRegionPoints then return None
    val depth = math.min(math.max(10, maxDepth), math.max(4, 2 * group.liberties.size + 2))
    val victimColor = group.color
    val original = group.stones
    attackerMoves(state, group, region, attacker).collectFirst(Function.unlift { move =>
      state.place(move, attacker).toOption
        .filter(next =>
          attackerWins(next, victimColor, original, region, attacker.opposite, attacker, depth - 1, budget)
        )
        .map(_ => move)
    })

  /** Shared node counter so a whole analysis stays inside its budget. */
  final class NodeBudget(val limit: Int):
    private var used = 0
    def exhausted: Boolean = used >= limit
    def visit(): Boolean =
      used += 1
      used <= limit

  // --------------------------------------------------------------------------
  // The search
  // --------------------------------------------------------------------------

  private def attackerWins(
      state: BoardState,
      victimColor: Color,
      original: Set[Point],
      region: Set[Point],
      toMove: Color,
      attacker: Color,
      depth: Int,
      budget: NodeBudget
  ): Boolean =
    // Every original stone gone means the group was captured (a partial
    // capture that leaves one original stone still falls short, correctly:
    // the group lives on).
    if original.forall(p => state.stoneAt(p) != victimColor) then true
    else if depth <= 0 || !budget.visit() then false
    else if toMove == attacker then
      // Move generation needs the victim's current group.  It is found
      // through a surviving original stone, never through a fixed point:
      // the head stone may already be gone while the group lives on.
      // (`survivor` always exists here: the all-gone case returned above.)
      val survivor = original.find(p => state.stoneAt(p) == victimColor).get
      val group = state.groupAt(survivor).get
      attackerMoves(state, group, region, attacker).exists { move =>
        state.place(move, attacker) match
          case Right(next) =>
            attackerWins(next, victimColor, original, region, attacker.opposite, attacker, depth - 1, budget)
          case Left(_) => false
      }
    else
      // The defender must have NO reply that saves the group: one refutation
      // is enough to fail the proof, which is what keeps this sound.
      defenderMoves(state, region, toMove).forall { move =>
        state.place(move, toMove) match
          case Right(next) =>
            attackerWins(next, victimColor, original, region, attacker, attacker, depth - 1, budget)
          case Left(_) => true // an illegal reply saves nobody
      } && defenderPassLoses(state, victimColor, original, region, attacker, depth, budget)

  private def defenderPassLoses(
      state: BoardState,
      victimColor: Color,
      original: Set[Point],
      region: Set[Point],
      attacker: Color,
      depth: Int,
      budget: NodeBudget
  ): Boolean =
    attackerWins(state.passFor(state.toMove), victimColor, original, region, attacker, attacker, depth - 1, budget)

  // --------------------------------------------------------------------------
  // Move generation: deliberately narrow
  // --------------------------------------------------------------------------

  /**
   * The stones, their liberties, and one ring beyond: everything a local
   * capture or rescue can touch.  Fixed for the whole proof, so both sides
   * reason about the same neighbourhood.
   */
  private[go] def localRegion(state: BoardState, victim: Group): Set[Point] =
    val out = mutable.Set.empty[Point] ++ victim.stones ++ victim.liberties
    for lib <- victim.liberties do
      out ++= state.neighborsOf(lib)
    out.toSet

  /**
   * Attacker tries the victim's liberties first (filling liberties is how
   * enclosed groups die), then any region move that captures something.
   * Anything else -- connecting far away, tenuki -- cannot force a local
   * capture, so leaving it out only risks missing, never misproving.
   */
  private def attackerMoves(
      state: BoardState,
      victim: Group,
      region: Set[Point],
      attacker: Color
  ): Vector[Point] =
    val libs = victim.liberties.toVector.filter(p => state.isLegal(p, attacker))
    val captures = region.toVector
      .filter(p => !victim.liberties.contains(p) && state.isLegal(p, attacker))
      .filter { p =>
        state.computeMove(p.index, attacker).toOption.exists(_.captured.nonEmpty)
      }
    // Deterministic order: lowest index first, so proofs reproduce exactly.
    (libs.sortBy(_.index) ++ captures.sortBy(_.index)).distinct

  /**
   * Defender tries everything nearby plus passing (handled separately): a
   * proof must beat every reply, so leaving one out would be unsound.
   * "Nearby" is any region point touching the victim's stones or liberties.
   */
  private def defenderMoves(state: BoardState, region: Set[Point], defender: Color): Vector[Point] =
    region.toVector.filter(p => state.isLegal(p, defender)).sortBy(_.index)
