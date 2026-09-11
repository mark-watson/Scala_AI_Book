#!/usr/bin/env python3
# Copyright 2026 Mark Watson. All rights reserved.
"""Read the self-play training data written by SelfPlay.scala.

The Scala engine writes one binary file containing every training example from
a self-play run.  This module reads it, can print a summary or draw a single
position, and can hand the whole set to NumPy for training.

Format (little-endian, written by DataOutputStream)
---------------------------------------------------
    magic         4 bytes   ASCII "GOPP"
    version       int32     currently 1
    exampleCount  int32
    planeCount    int32     15 for the current feature encoding
    boardSize     int32     9, 13 or 19
    policySize    int32     boardSize * boardSize + 1 (pass is last)
    then, per example, tightly packed float32:
      planeCount * boardSize * boardSize   feature planes
      policySize                           MCTS visit distribution
      1                                    outcome z
      1                                    root search value

Every record is the same size, so the body is a single flat float32 block that
reshapes directly.  The summary and the drawing use only the standard library;
NumPy is needed only to feed a training loop.

Usage
-----
    python3 load_selfplay.py selfplay-9x9/training.bin
    python3 load_selfplay.py selfplay-9x9/training.bin --example 40
    python3 load_selfplay.py selfplay-9x9/training.bin --npz training.npz

Using the data in PyTorch
-------------------------
    import torch
    from load_selfplay import read_training_data

    data = read_training_data("selfplay-9x9/training.bin")
    planes = torch.from_numpy(data["planes"])             # (N, 15, size, size)
    target_policy = torch.from_numpy(data["policies"])    # (N, size*size+1)
    target_value = torch.from_numpy(data["outcomes"])     # (N,) in {-1, 0, +1}

    # AlphaGo Zero loss: cross-entropy against the search policy plus a
    # mean-squared error against the game result.
    log_policy, value = model(planes)
    loss = -(target_policy * log_policy).sum(dim=1).mean() + \
           ((value - target_value) ** 2).mean()

The data is written from the point of view of the player to move in each
position, so the model learns a mover-perspective value exactly as the engine
evaluates it.
"""

from __future__ import annotations

import argparse
import array
import struct
import sys
from dataclasses import dataclass

MAGIC = b"GOPP"
HEADER = "<iiiii"  # version, exampleCount, planeCount, boardSize, policySize
HEADER_SIZE = 4 + struct.calcsize(HEADER)
COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # GTP column letters, I is skipped

try:
    import numpy as np
except ImportError:  # NumPy is optional; only the training API needs it
    np = None


class SelfPlayFormatError(Exception):
    """Raised when a file is not a self-play training file."""


@dataclass
class Header:
    version: int
    example_count: int
    plane_count: int
    board_size: int
    policy_size: int

    @property
    def plane_floats(self) -> int:
        return self.plane_count * self.board_size * self.board_size

    @property
    def floats_per_example(self) -> int:
        return self.plane_floats + self.policy_size + 2

    def describe(self) -> str:
        return (
            f"version {self.version}, {self.example_count} examples, "
            f"{self.plane_count} planes of {self.board_size}x{self.board_size}, "
            f"policy of {self.policy_size} moves "
            f"({self.floats_per_example} floats per example)"
        )


def read_header(path: str) -> Header:
    """Reads and validates only the header."""
    with open(path, "rb") as handle:
        magic = handle.read(4)
        if magic != MAGIC:
            raise SelfPlayFormatError(
                f"{path}: expected magic {MAGIC!r}, found {magic!r}; "
                "this is not a self-play training file"
            )
        raw = handle.read(struct.calcsize(HEADER))
        if len(raw) != struct.calcsize(HEADER):
            raise SelfPlayFormatError(f"{path}: truncated header")
        header = Header(*struct.unpack(HEADER, raw))

    if header.version != 1:
        raise SelfPlayFormatError(f"{path}: unsupported version {header.version}")
    if header.board_size <= 0 or header.plane_count <= 0:
        raise SelfPlayFormatError(f"{path}: nonsensical header {header}")
    expected_policy = header.board_size * header.board_size + 1
    if header.policy_size != expected_policy:
        raise SelfPlayFormatError(
            f"{path}: policy size {header.policy_size} does not match "
            f"{header.board_size}x{header.board_size} + pass"
        )
    return header


def read_floats(path: str) -> tuple[Header, array.array]:
    """Reads the header and the flat float32 body, using only the stdlib."""
    header = read_header(path)
    expected = header.example_count * header.floats_per_example
    body = array.array("f")
    with open(path, "rb") as handle:
        handle.seek(HEADER_SIZE)
        try:
            body.fromfile(handle, expected)
        except EOFError:
            raise SelfPlayFormatError(
                f"{path}: truncated body, expected {expected} floats"
            ) from None
    # The file is little-endian; array('f') is native, so swap on big-endian.
    if sys.byteorder == "big":
        body.byteswap()
    if len(body) != expected:
        raise SelfPlayFormatError(f"{path}: expected {expected} floats, found {len(body)}")
    return header, body


def read_training_data(path: str) -> dict:
    """Reads a training file into NumPy arrays, ready for a training loop.

    Returns a dict with keys:
      planes        (N, planeCount, boardSize, boardSize) float32
      policies      (N, policySize) float32, each summing to 1
      outcomes      (N,) float32, each in {-1, 0, +1}
      search_values (N,) float32, the root win rate when the move was chosen
      header        the Header
    """
    if np is None:
        raise RuntimeError("NumPy is required for this function: pip install numpy")

    header, body = read_floats(path)
    records = np.frombuffer(body, dtype="<f4").reshape(
        header.example_count, header.floats_per_example
    )
    planes = records[:, : header.plane_floats].reshape(
        header.example_count, header.plane_count, header.board_size, header.board_size
    )
    policies = records[:, header.plane_floats : header.plane_floats + header.policy_size]
    return {
        "planes": planes.copy(),
        "policies": policies.copy(),
        "outcomes": records[:, -2].copy(),
        "search_values": records[:, -1].copy(),
        "header": header,
    }


def _record(header: Header, body: array.array, index: int) -> memoryview:
    start = index * header.floats_per_example
    return memoryview(body)[start : start + header.floats_per_example]


def vertex(index: int, size: int) -> str:
    """A policy index as a GTP vertex; the last index is the pass move."""
    if index == size * size:
        return "pass"
    x, y = index % size, index // size
    return f"{COLUMNS[x]}{size - y}"


def summarize(header: Header, body: array.array) -> str:
    """A human-readable summary, computed without NumPy."""
    floats_per_example = header.floats_per_example
    policy_start = header.plane_floats
    policy_end = policy_start + header.policy_size

    wins = losses = draws = 0
    policy_min = float("inf")
    policy_max = float("-inf")
    value_total = 0.0
    plane_means = [0.0] * header.plane_count

    for index in range(header.example_count):
        record = _record(header, body, index)
        outcome = record[-2]
        if outcome > 0:
            wins += 1
        elif outcome < 0:
            losses += 1
        else:
            draws += 1
        policy_sum = sum(record[policy_start:policy_end])
        policy_min = min(policy_min, policy_sum)
        policy_max = max(policy_max, policy_sum)
        value_total += record[-1]
        for plane in range(header.plane_count):
            start = plane * header.board_size * header.board_size
            plane_means[plane] += sum(
                record[start : start + header.board_size * header.board_size]
            )

    count = max(1, header.example_count)
    lines = [
        f"examples       {header.example_count}",
        f"planes         {header.plane_count} x {header.board_size} x {header.board_size}",
        f"policy         {header.policy_size} moves (pass is the last index)",
        f"outcomes       {wins} wins, {losses} losses, {draws} draws",
        f"policy sums    min {policy_min:.4f}, max {policy_max:.4f}",
        f"search values  mean {value_total / count:+.4f}",
        "",
        "mean activation per plane:",
    ]
    for plane in range(header.plane_count):
        lines.append(f"  plane {plane:2d}  {plane_means[plane] / count:8.2f}")
    return "\n".join(lines)


def render_example(header: Header, body: array.array, index: int) -> str:
    """Draws one position as a text board, with its search target."""
    size = header.board_size
    if index < 0 or index >= header.example_count:
        raise IndexError(f"example {index} is out of range 0..{header.example_count - 1}")

    record = _record(header, body, index)
    square = size * size
    me = record[0:square]                # plane 0: the player to move
    them = record[square : 2 * square]   # plane 1: the opponent
    side_to_move = record[13 * square : 14 * square]  # plane 13: 1 if Black
    black_to_move = sum(side_to_move) > square / 2.0

    rows = []
    for y in range(size):
        cells = []
        for x in range(size):
            i = y * size + x
            if me[i] > 0.5:
                cells.append("X" if black_to_move else "O")
            elif them[i] > 0.5:
                cells.append("O" if black_to_move else "X")
            else:
                cells.append(".")
        rows.append(" ".join(cells))

    policy = record[header.plane_floats : header.plane_floats + header.policy_size]
    ranked = sorted(range(header.policy_size), key=lambda i: -policy[i])[:6]

    lines = [
        f"example {index}: {'Black' if black_to_move else 'White'} to move, "
        f"game outcome z = {record[-2]:+.0f}, search value = {record[-1]:+.3f}",
        "",
    ]
    lines += ["   " + row for row in rows]
    lines.append("")
    lines.append("  most visited moves in the search target:")
    for i in ranked:
        lines.append(f"    {vertex(i, size):<5} {policy[i] * 100:5.1f}%")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read and inspect self-play training data written by SelfPlay.scala."
    )
    parser.add_argument("path", help="path to training.bin")
    parser.add_argument("--example", type=int, default=None,
                        help="draw one example instead of printing a summary")
    parser.add_argument("--npz", default=None,
                        help="also save the arrays to a compressed .npz file (needs NumPy)")
    args = parser.parse_args(argv)

    try:
        header, body = read_floats(args.path)
    except (SelfPlayFormatError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"{args.path}: {header.describe()}")
    print()
    if args.example is not None:
        try:
            print(render_example(header, body, args.example))
        except IndexError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    else:
        print(summarize(header, body))

    if args.npz:
        if np is None:
            print("error: --npz needs NumPy: pip install numpy", file=sys.stderr)
            return 2
        data = read_training_data(args.path)
        np.savez_compressed(
            args.npz,
            planes=data["planes"],
            policies=data["policies"],
            outcomes=data["outcomes"],
            search_values=data["search_values"],
        )
        print()
        print(f"saved {args.npz}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
