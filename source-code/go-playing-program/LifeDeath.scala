//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import scala.collection.mutable

// ============================================================================
// LifeDeath.scala -- life-and-death analysis for scoring and GTP status lists.
//
// Dead stones left on the board poison the count: a group that cannot
// possibly live still owns its points under a naive flood fill.  This file
// decides, for every group, whether it is alive, dead, or seki (mutually
// alive), in three tiers of increasing cost:
//
//   1. Benson unconditional life: groups with two vital eyes are alive, no
//      search needed.  Sound.
//   2. Territory control: a small enclosed group with fewer than two vital
//      eyes, whose every liberty is owned by the opponent once the group
//      itself is lifted off, is dead.  Cheap floods, no search -- but a
//      heuristic, so a group is spared when it shares a liberty with a small
//      opposing group that is itself unsettled.  That shared case is seki
//      more often than it is a double oversight.
//   3. Tactical proof: a non-alive group the opponent can force-capture
//      ([[Tactics.canForceCapture]]) is dead.  Sound when it fires; it
//      declines open positions rather than guessing.
//
// What `score` uses versus what the status commands use: the per-rollout
// scoring path in Board.scala runs the same tiers but proves tactically only
// for groups with two or fewer liberties, and only when they are enclosed;
// the GTP `final_status_list` path proves for groups up to six liberties
// anywhere.  The two can disagree on an open fight left on the board at
// pass-pass; the status answer is the more careful one.
// ============================================================================

object LifeDeath:
  /** Small-group cutoff: at most this many stones still counts as "small". */
  val SmallGroupStones = 10

  /** Tactical proofs in the scoring path run only up to this many liberties. */
  val ScorePathMaxLiberties = 2

  /** Tactical proofs in the status path run up to this many liberties. */
  val StatusPathMaxLiberties = 6

  /** Outcome of [[analyze]], as stone sets. */
  final case class Verdict(alive: Set[Point], dead: Set[Point], seki: Set[Point])

  /** Full analysis: every stone lands in exactly one of alive, dead or seki. */
  def analyze(state: BoardState): Verdict =
    val groups = state.groups.values.toVector.distinct
    val alive = bensonAlive(state, groups)
    val aliveSet = alive.map(_.stones).foldLeft(Set.empty[Point])(_ ++ _)
    val deadGroups = groups.filter(g =>
      !alive.contains(g) && isDead(state, g, groups, alive, StatusPathMaxLiberties, false)
    )
    val deadSet = deadGroups.map(_.stones).foldLeft(Set.empty[Point])(_ ++ _)
    val rest = groups.filter(g => !alive.contains(g) && !deadGroups.contains(g))
    val sekiGroups = rest.filter(g => isSeki(state, g, rest))
    val sekiSet = sekiGroups.map(_.stones).foldLeft(Set.empty[Point])(_ ++ _)
    Verdict(aliveSet, deadSet, sekiSet)

  /** Groups with two or more vital eyes. */
  def aliveGroups(state: BoardState): Vector[Group] =
    bensonAlive(state, state.groups.values.toVector.distinct).toVector

  /** Groups the opponent can force-capture, or that own nothing. */
  def deadGroups(state: BoardState): Vector[Group] =
    val groups = state.groups.values.toVector.distinct
    val alive = bensonAlive(state, groups)
    groups.filter(g => !alive.contains(g) && isDead(state, g, groups, alive, StatusPathMaxLiberties, false))

  /** Mutually-alive neighbouring groups; see [[isSeki]]. */
  def sekiGroups(state: BoardState): Vector[Group] =
    val groups = state.groups.values.toVector.distinct
    val alive = bensonAlive(state, groups)
    val dead =
      groups.filter(g => !alive.contains(g) && isDead(state, g, groups, alive, StatusPathMaxLiberties, false))
    val rest = groups.filter(g => !alive.contains(g) && !dead.contains(g))
    rest.filter(g => isSeki(state, g, rest))

  /** Every dead stone on the board. */
  def deadStones(state: BoardState): Set[Point] =
    deadGroups(state).map(_.stones).foldLeft(Set.empty[Point])(_ ++ _)

  /**
   * Cheap gate for the scoring path: true when any group is short enough on
   * liberties that the count might be wrong.  Settled positions skip the
   * analysis entirely and score exactly as before.
   */
  def needsCheck(state: BoardState): Boolean =
    state.groups.values.toVector.distinct.exists(_.liberties.size <= 6)

  /**
   * Dead stones for the hot scoring path: Benson plus tactical proofs for
   * tiny groups plus the territory rule.  Cheaper than [[deadStones]] and
   * exact on every enclosed position; an open fight left on the board at
   * pass-pass is counted as it stands.
   */
  def deadStonesForScore(state: BoardState): Set[Point] =
    val groups = state.groups.values.toVector.distinct
    if !groups.exists(_.liberties.size <= 6) then Set.empty
    else
      val alive = bensonAlive(state, groups)
      groups
        .filter(g => !alive.contains(g) && isDead(state, g, groups, alive, ScorePathMaxLiberties, true))
        .map(_.stones)
        .foldLeft(Set.empty[Point])(_ ++ _)

  // --------------------------------------------------------------------------
  // Tier 1: Benson unconditional life
  // --------------------------------------------------------------------------

  /**
   * Groups with at least two vital eyes, by Benson's algorithm: start with
   * every group alive, then repeatedly retire groups with fewer than two eyes
   * whose neighbouring stones are all still alive.  What survives the fixpoint
   * is unconditionally alive -- the opponent cannot touch it whatever happens
   * elsewhere.
   */
  private[go] def bensonAlive(state: BoardState, groups: Vector[Group]): Set[Group] =
    val byStone = state.groups
    val alive = mutable.Set.empty[Group] ++ groups
    var changed = true
    while changed do
      changed = false
      for group <- alive.toVector do
        val vital = candidateEyes(state, group).count(eye =>
          state.neighborsOf(eye).forall(q =>
            byStone.get(q).exists(nb => nb.color == group.color && alive.contains(nb))
          )
        )
        if vital < 2 then
          alive -= group
          changed = true
    alive.toSet

  /**
   * Candidate one-point eyes: liberties whose every neighbour is already this
   * group's colour.  Edge and corner points count with their fewer neighbours,
   * which is what makes corner life work.
   */
  private[go] def candidateEyes(state: BoardState, group: Group): Vector[Point] =
    group.liberties.toVector.filter { eye =>
      val nbrs = state.neighborsOf(eye)
      nbrs.nonEmpty && nbrs.forall(q => state.stoneAt(q) == group.color)
    }

  // --------------------------------------------------------------------------
  // Tier 2 + 3: death
  // --------------------------------------------------------------------------

  /**
   * The territory rule runs first because it is cheap: most genuinely dead
   * groups own nothing, and proving that needs floods, not search.  The
   * tactical proof then covers what the rule cannot see -- small groups left
   * in atari in open positions.  The scoring path additionally requires such
   * groups to be enclosed, so an open fight left on the board at pass-pass is
   * counted as it stands rather than searched on a hot path.
   */
  private def isDead(
      state: BoardState,
      group: Group,
      groups: Vector[Group],
      alive: Set[Group],
      tacticalLiberties: Int,
      requireEnclosed: Boolean
  ): Boolean =
    if territoryDead(state, group, groups, alive, tacticalLiberties, requireEnclosed) then true
    else searchDead(state, group, tacticalLiberties, requireEnclosed)

  private def searchDead(
      state: BoardState,
      group: Group,
      tacticalLiberties: Int,
      requireEnclosed: Boolean
  ): Boolean =
    group.liberties.size <= tacticalLiberties
      && group.liberties.size <= Tactics.MaxVictimLiberties
      && (!requireEnclosed || enclosedFast(state, group))
      && Tactics.localRegion(state, group).size <= Tactics.MaxRegionPoints
      && Tactics.canForceCapture(state, group.stones.head, group.color.opposite)

  /**
   * True when every liberty is hemmed in: none has an empty neighbour outside
   * the group.  A group with an escape route is an open fight, not a scoring
   * detail, and the scoring path leaves it alone.
   */
  private[go] def enclosedFast(state: BoardState, group: Group): Boolean =
    group.liberties.forall(lib =>
      state.neighborsOf(lib).forall(q => !state.isEmptyAt(q) || group.stones.contains(q))
    )

  /**
   * Tier 3: with the group lifted off, every one of its liberties is owned by
   * the opponent.  The group surrounds nothing and can surround nothing, so
   * it is dead -- unless it shares a liberty with a small opposing group that
   * is itself unsettled, in which case the two may be sharing life (seki) and
   * both are spared.  Only small groups are ever candidates here: a wall
   * enclosing a prisoner is not sharing life with it, and a big group is
   * either Benson-alive or an open fight this rule cannot judge.
   */
  private[go] def territoryDead(
      state: BoardState,
      group: Group,
      groups: Vector[Group],
      alive: Set[Group],
      tacticalLiberties: Int,
      requireEnclosed: Boolean
  ): Boolean =
    if group.size > SmallGroupStones then false
    else if candidateEyes(state, group).size >= 2 then false
    else if !controlsAll(state, group) then false
    else
      !groups.exists(o =>
        o.color == group.color.opposite
          && o.size <= SmallGroupStones
          && !alive.contains(o)
          && o.liberties.exists(group.liberties.contains)
          && !searchDead(state, o, tacticalLiberties, requireEnclosed)
      )

  /**
   * With `group` lifted off the board, every one of its liberties is owned by
   * the opponent.
   */
  private def controlsAll(state: BoardState, group: Group): Boolean =
    if candidateEyes(state, group).size >= 2 then false
    else
      val without = state.cells.clone()
      for stone <- group.stones do without(stone.index) = 0
      val owner = BoardState.territoryOf(without, state.size)
      val attackerByte = group.color.opposite.ordinal.toByte
      group.liberties.forall(p => owner(p.index) == attackerByte)

  // --------------------------------------------------------------------------
  // Seki: neighbouring unsettled groups that share a liberty
  // --------------------------------------------------------------------------

  /**
   * Two opposing groups, neither alive nor dead, sharing at least one liberty:
   * neither side can force a capture (that is what "not dead" proved), so the
   * shared liberty is dame they both need.  Called only for groups outside
   * the alive and dead sets.
   */
  private def isSeki(state: BoardState, group: Group, rest: Vector[Group]): Boolean =
    rest.exists(o =>
      o.color == group.color.opposite && o.liberties.exists(group.liberties.contains)
    )
