//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import scala.concurrent.duration.FiniteDuration

/**
 * A tiny `--name value` argument parser shared by this project's entry points.
 *
 * Small enough not to justify a dependency, and deliberately forgiving:
 * `--size 9`, `--size=9` and `-size 9` all work, and a following token that
 * begins with `-` is treated as the next option rather than as a value.
 */
object Args:
  /** The result of parsing an argument vector. */
  final case class Parsed(
      options: Map[String, String],
      flags: Set[String],
      positional: Vector[String]
  ):
    def has(name: String): Boolean = flags.contains(name) || options.contains(name)

    def get(name: String): Option[String] = options.get(name)

    def string(name: String, default: String): String = options.getOrElse(name, default)

    def int(name: String, default: Int): Int =
      options.get(name).flatMap(_.toIntOption).getOrElse(default)

    def double(name: String, default: Double): Double =
      options.get(name).flatMap(_.toDoubleOption).getOrElse(default)

    def bool(name: String, default: Boolean): Boolean =
      if flags.contains(name) then true
      else options.get(name) match
        case Some(v) =>
          v.trim.toLowerCase match
            case "true" | "yes" | "on" | "1"  => true
            case "false" | "no" | "off" | "0" => false
            case _                            => default
        case None => default

    /** Parses `5s`, `250ms`, `2m` or a plain number of seconds. */
    def duration(name: String): Option[FiniteDuration] =
      options.get(name).flatMap(Args.parseDuration)

    /** Every option name that was supplied, for diagnostics. */
    def names: Set[String] = options.keySet ++ flags

  def parse(argv: Array[String]): Parsed =
    val options = scala.collection.mutable.Map.empty[String, String]
    val flags = scala.collection.mutable.Set.empty[String]
    val positional = Vector.newBuilder[String]
    var i = 0
    while i < argv.length do
      val arg = argv(i)
      val stripped = if arg.startsWith("--") then arg.drop(2) else if arg.startsWith("-") then arg.drop(1) else ""
      if stripped.nonEmpty then
        val eq = stripped.indexOf('=')
        if eq >= 0 then
          options(stripped.substring(0, eq)) = stripped.substring(eq + 1)
        else
          val next = if i + 1 < argv.length then Some(argv(i + 1)) else None
          next match
            case Some(v) if !v.startsWith("-") =>
              options(stripped) = v
              i += 1
            case _ =>
              flags += stripped
      else positional += arg
      i += 1
    Parsed(options.toMap, flags.toSet, positional.result())

  /** Parses a duration like `250ms`, `5s`, `2m`, `1h` or a bare second count. */
  def parseDuration(text: String): Option[FiniteDuration] =
    val t = text.trim.toLowerCase
    // "ms" must be tested before "s", which it also ends with.
    val (numberPart, millisPerUnit) =
      if t.endsWith("ms") then (t.dropRight(2), 1.0)
      else if t.endsWith("s") then (t.dropRight(1), 1000.0)
      else if t.endsWith("m") then (t.dropRight(1), 60_000.0)
      else if t.endsWith("h") then (t.dropRight(1), 3_600_000.0)
      else (t, 1000.0)
    numberPart.toDoubleOption
      .filter(_ >= 0)
      .map(n => FiniteDuration((n * millisPerUnit).round, "ms"))
