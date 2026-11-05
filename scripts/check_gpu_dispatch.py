#!/usr/bin/env python3

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

CASE_RE = re.compile(r"case\s+([A-Za-z_]\w*)\s*:")
DEFAULT_RE = re.compile(r"default\s*:")
MACRO_ENTRY_RE = re.compile(r"\bX\s*\(\s*(F_[A-Za-z0-9_]+)\s*,\s*(V_[A-Za-z0-9_]+)\s*\)")
RUNOP_RE = re.compile(r"\bvoid\s+RunOp\s*\(")
SWITCH_RE = re.compile(r"\bswitch\s*\(")


class CheckError(Exception):
    pass


@dataclass(frozen=True)
class Entry:
    name: str
    line: int


def mask_comments_and_literals(text: str) -> str:
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("//", i):
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif text.startswith("/*", i):
            out[i] = " "
            out[i + 1] = " "
            i += 2
            while i < n and not text.startswith("*/", i):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                out[i + 1] = " "
                i += 2
        elif text[i] in ('"', "'"):
            quote = text[i]
            out[i] = " "
            i += 1
            while i < n:
                ch = text[i]
                if ch == "\\":
                    out[i] = " "
                    if i + 1 < n:
                        if text[i + 1] != "\n":
                            out[i + 1] = " "
                        i += 2
                    else:
                        i += 1
                else:
                    if ch != "\n":
                        out[i] = " "
                    i += 1
                    if ch == quote:
                        break
        else:
            i += 1
    return "".join(out)


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def is_boundary(text: str, pos: int) -> bool:
    return pos == 0 or not (text[pos - 1].isalnum() or text[pos - 1] == "_")


def parse_header(path: Path) -> list[Entry]:
    text = mask_comments_and_literals(path.read_text(encoding="utf-8"))
    entries = [Entry(m.group(1), line_of(text, m.start())) for m in MACRO_ENTRY_RE.finditer(text)]
    if not entries:
        raise CheckError(f"no GPU VM op definitions found in {path}")
    return entries


def find_runop_switch(text: str) -> int:
    match = RUNOP_RE.search(text)
    if not match:
        raise CheckError("RunOp definition not found")

    match = SWITCH_RE.search(text, match.end())
    if not match:
        raise CheckError("switch statement not found in RunOp")

    brace = text.find("{", match.end())
    if brace < 0:
        raise CheckError("switch body not found in RunOp")
    return brace


def parse_runop_cases(path: Path) -> tuple[list[Entry], int | None]:
    text = mask_comments_and_literals(path.read_text(encoding="utf-8"))
    start = find_runop_switch(text)
    entries: list[Entry] = []
    default_line = None
    depth = 0
    i = start
    found_body = False

    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            found_body = True
            i += 1
            continue
        if ch == "}":
            depth -= 1
            if depth < 0:
                raise CheckError("unbalanced braces in RunOp switch")
            i += 1
            if depth == 0:
                break
            continue
        if depth == 1:
            match = CASE_RE.match(text, i)
            if match and is_boundary(text, i):
                entries.append(Entry(match.group(1), line_of(text, i)))
                i = match.end()
                continue
            match = DEFAULT_RE.match(text, i)
            if match and is_boundary(text, i):
                if default_line is None:
                    default_line = line_of(text, i)
                i = match.end()
                continue
        i += 1

    if not found_body or depth != 0:
        raise CheckError("unterminated RunOp switch body")
    if not entries:
        raise CheckError("no case labels found in RunOp switch")
    return entries, default_line


def validate(expected: list[Entry], actual: list[Entry], default_line: int | None, path: Path) -> list[str]:
    errors: list[str] = []
    expected_names = [entry.name for entry in expected]
    actual_names = [entry.name for entry in actual]

    lines_by_name: dict[str, list[int]] = {}
    for entry in actual:
        lines_by_name.setdefault(entry.name, []).append(entry.line)
    duplicates = {name: lines for name, lines in lines_by_name.items() if len(lines) > 1}
    for name, lines in sorted(duplicates.items()):
        errors.append(f"duplicate case {name} at {path} lines {', '.join(map(str, lines))}")

    expected_set = set(expected_names)
    actual_set = set(actual_names)
    for entry in expected:
        if entry.name not in actual_set:
            errors.append(f"missing case {entry.name}, defined at header line {entry.line}")
    for entry in actual:
        if entry.name not in expected_set:
            errors.append(f"unexpected case {entry.name} at {path} line {entry.line}")

    if not duplicates and actual_set == expected_set and actual_names != expected_names:
        for index, (expected_entry, actual_entry) in enumerate(zip(expected, actual)):
            if expected_entry.name != actual_entry.name:
                errors.append(
                    "order mismatch at index "
                    f"{index}: expected {expected_entry.name} (header line {expected_entry.line}), "
                    f"found {actual_entry.name} ({path} line {actual_entry.line})"
                )
                break

    if default_line is None:
        errors.append(f"missing default label in RunOp switch ({path})")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that RunOp switch cases match the GPU VM op id order exactly."
    )
    parser.add_argument("header", nargs="?", default="src/vm_gpu.h")
    parser.add_argument("cuda", nargs="?", default="src/vm_gpu.cu")
    args = parser.parse_args(argv)

    header_path = Path(args.header)
    cuda_path = Path(args.cuda)

    try:
        expected = parse_header(header_path)
        actual, default_line = parse_runop_cases(cuda_path)
    except (CheckError, OSError) as error:
        print(f"GPU RunOp switch check failed: {error}", file=sys.stderr)
        return 1

    errors = validate(expected, actual, default_line, cuda_path)
    if errors:
        print("GPU RunOp switch check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
