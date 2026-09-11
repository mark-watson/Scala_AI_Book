#!/usr/bin/env python3
# Copyright 2026 Mark Watson. All rights reserved.
"""Tests for load_selfplay.py, the self-play training-data reader.

Stdlib only, so this runs anywhere Python does (NumPy-dependent paths are
skipped when NumPy is missing):

    cd python && python3 -m unittest
    # or from the repository root:
    make test-python
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest

import load_selfplay
from load_selfplay import (
    SelfPlayFormatError,
    read_floats,
    read_header,
    render_example,
    summarize,
    vertex,
)


def write_training_bin(path, board_size=9, plane_count=15, count=3,
                       policy_sums_to_one=True):
    """Writes a reader-compatible file with `count` synthetic examples."""
    policy_size = board_size * board_size + 1
    with open(path, "wb") as handle:
        handle.write(b"GOPP")
        handle.write(struct.pack("<iiiii", 1, count, plane_count,
                                 board_size, policy_size))
        for index in range(count):
            planes = [0.0] * (plane_count * board_size * board_size)
            planes[index % len(planes)] = 1.0  # one stone per example
            handle.write(struct.pack("<%df" % len(planes), *planes))
            if policy_sums_to_one:
                policy = [0.0] * policy_size
                policy[index % policy_size] = 1.0
            else:
                policy = [0.5] * policy_size
            handle.write(struct.pack("<%df" % len(policy), *policy))
            handle.write(struct.pack("<f", 1.0 if index % 2 == 0 else -1.0))
            handle.write(struct.pack("<f", 0.25))


class ReaderTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "training.bin")
        write_training_bin(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_header_round_trips(self):
        header = read_header(self.path)
        self.assertEqual(header.version, 1)
        self.assertEqual(header.example_count, 3)
        self.assertEqual(header.plane_count, 15)
        self.assertEqual(header.board_size, 9)
        self.assertEqual(header.policy_size, 82)

    def test_body_has_the_documented_shape(self):
        header, body = read_floats(self.path)
        self.assertEqual(len(body),
                         header.example_count * header.floats_per_example)
        self.assertIn("3 examples", header.describe())
        self.assertIn("15 planes", header.describe())

    def test_policies_sum_to_one(self):
        header, body = read_floats(self.path)
        report = summarize(header, body)
        self.assertIn("min 1.0000, max 1.0000", report)

    def test_outcomes_are_counted(self):
        header, body = read_floats(self.path)
        report = summarize(header, body)
        self.assertIn("2 wins, 1 losses, 0 draws", report)

    def test_example_renders(self):
        header, body = read_floats(self.path)
        drawing = render_example(header, body, 0)
        self.assertIn("example 0", drawing)
        self.assertIn("game outcome z", drawing)

    def test_example_out_of_range_fails_loudly(self):
        header, body = read_floats(self.path)
        with self.assertRaises(IndexError):
            render_example(header, body, 99)

    def test_vertex_names_the_pass_move(self):
        self.assertEqual(vertex(81, 9), "pass")
        self.assertEqual(vertex(0, 9), "A9")

    def test_numpy_path_matches_the_stdlib_path(self):
        if load_selfplay.np is None:
            self.skipTest("NumPy is not installed")
        data = load_selfplay.read_training_data(self.path)
        self.assertEqual(data["planes"].shape, (3, 15, 9, 9))
        self.assertEqual(data["policies"].shape, (3, 82))
        for row in data["policies"]:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=5)
        self.assertEqual(list(data["outcomes"]), [1.0, -1.0, 1.0])


class CorruptFileTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def file(self, name="training.bin"):
        return os.path.join(self.dir.name, name)

    def test_truncated_body_fails_loudly(self):
        # A header promising three examples with only one example of bytes.
        path = self.file()
        write_training_bin(path, count=3)
        with open(path, "r+b") as handle:
            handle.truncate(4 + 20 + 1 * (15 * 81 + 82 + 2) * 4)
        with self.assertRaises(SelfPlayFormatError):
            read_floats(path)

    def test_truncated_header_fails_loudly(self):
        path = self.file()
        with open(path, "wb") as handle:
            handle.write(b"GOPP\x01")
        with self.assertRaises(SelfPlayFormatError):
            read_header(path)

    def test_wrong_magic_fails_loudly(self):
        path = self.file()
        with open(path, "wb") as handle:
            handle.write(b"NOPE" + struct.pack("<iiiii", 1, 1, 15, 9, 82))
        with self.assertRaises(SelfPlayFormatError):
            read_header(path)

    def test_wrong_policy_size_fails_loudly(self):
        path = self.file()
        with open(path, "wb") as handle:
            handle.write(b"GOPP" + struct.pack("<iiiii", 1, 1, 15, 9, 7))
        with self.assertRaises(SelfPlayFormatError):
            read_header(path)

    def test_missing_file_raises_os_error(self):
        with self.assertRaises(OSError):
            read_header(self.file("absent.bin"))


if __name__ == "__main__":
    sys.exit(unittest.main())
