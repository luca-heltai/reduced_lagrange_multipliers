#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Publish or check the SVG/CSV outputs of LocalCoRefinement.DocumentationExamples."""

import argparse
import csv
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET


NAMES = ("space", "embedded", "both", "both-wide", "both-post", "both-capped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_output", type=Path, help="Build-tree examples directory")
    parser.add_argument("--check", action="store_true", help="Compare without changing assets")
    args = parser.parse_args()
    source = args.test_output.resolve()
    root = Path(__file__).resolve().parents[1]
    destination = root / "doc" / "_static" / "local-co-refinement"
    if source == destination.resolve():
        parser.error("Use fresh build-tree test outputs, not the published assets")

    with (source / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if tuple(row["example"] for row in rows) != NAMES:
        parser.error("The summary does not contain the six expected test cases")
    for name in NAMES:
        ET.parse(source / f"{name}.svg")

    filenames = [f"{name}.svg" for name in NAMES] + ["summary.csv"]
    if args.check:
        different = [name for name in filenames
                     if not (destination / name).is_file()
                     or (source / name).read_bytes() != (destination / name).read_bytes()]
        if different:
            print("Published assets differ: " + ", ".join(different), file=sys.stderr)
            return 1
        print("Published figures match the test outputs")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            shutil.copyfile(source / name, destination / name)
        print(f"Published {len(filenames)} test outputs in {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
