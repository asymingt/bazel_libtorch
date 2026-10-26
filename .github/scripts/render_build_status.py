#!/usr/bin/env python3
"""Renders the host x target build matrix of a Pull Request workflow run.

Usage: render_build_status.py JOBS_JSONL OUT_DIR RUN_URL COMMIT BAZEL STEM

JOBS_JSONL holds one GitHub Actions job object per line (from
`gh api .../actions/runs/<id>/jobs --paginate --jq '.jobs[]'`). Writes
OUT_DIR/STEM.svg (the matrix shown in the README) and OUT_DIR/STEM.json (the
same data, machine readable).

Only the jobs of the `bazel BAZEL` stage (e.g. `9.x`, `last_rc`) are
considered. Diagonal cells are the native jobs (build + test); off-diagonal
cells are cross builds (host -> target).
"""

import html
import json
import re
import sys

PLATFORMS = [
    "linux-amd64",
    "linux-arm64",
    "darwin-amd64",
    "darwin-arm64",
    "windows-amd64",
    "windows-arm64",
]

# conclusion -> (symbol, fill colour, legend text)
STYLES = {
    "success": ("✓", "#2da44e", "pass"),
    "failure": ("✗", "#cf222e", "fail"),
    "cancelled": ("–", "#8c959f", "cancelled"),
    "skipped": ("–", "#8c959f", "skipped (dependency failed)"),
    "pending": ("…", "#bf8700", "running"),
}


def outcome(job):
    if job.get("status") != "completed":
        return "pending"
    conclusion = job.get("conclusion") or "pending"
    return conclusion if conclusion in STYLES else "failure"


def collect(jobs, bazel):
    suffix = rf" / bazel {re.escape(bazel)}(?: |$)"
    native_re = re.compile(r"^native (\S+)" + suffix)
    cross_re = re.compile(r"^cross (\S+) -> (\S+)" + suffix)
    cells = {}
    for job in jobs:
        name = job.get("name", "")
        m = native_re.match(name)
        if m:
            cells[(m.group(1), m.group(1))] = outcome(job)
            continue
        m = cross_re.match(name)
        if m:
            cells[(m.group(1), m.group(2))] = outcome(job)
    return cells


def render_svg(cells, commit, bazel):
    cell, label_w, header_h, pad = 34, 118, 92, 8
    n = len(PLATFORMS)
    legend = ("success", "failure", "skipped", "pending")
    legend_w = sum(25 + 7 * len(STYLES[s][2]) for s in legend)
    # Rotated column headers overhang the last column by ~50px.
    width = max(label_w + n * cell + 56, pad + legend_w + pad)
    legend_y = header_h + n * cell + 22
    height = legend_y + 40
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad}" y="16" font-weight="600">host ↓ / target →</text>',
    ]
    for i, target in enumerate(PLATFORMS):
        x = label_w + i * cell + cell / 2
        out.append(
            f'<text transform="translate({x + 4},{header_h - 6}) rotate(-60)" '
            f'fill="#24292f">{target}</text>'
        )
    for r, host in enumerate(PLATFORMS):
        y = header_h + r * cell
        out.append(
            f'<text x="{label_w - 8}" y="{y + cell / 2 + 4}" text-anchor="end" '
            f'fill="#24292f">{host}</text>'
        )
        for c, target in enumerate(PLATFORMS):
            x = label_w + c * cell
            state = cells.get((host, target), "pending")
            symbol, colour, _ = STYLES[state]
            kind = "native build + test" if host == target else "cross build"
            out.append(
                f'<g><title>{html.escape(host)} → {html.escape(target)}: '
                f'{kind}, {state}</title>'
                f'<rect x="{x + 2}" y="{y + 2}" width="{cell - 4}" height="{cell - 4}" '
                f'rx="4" fill="{colour}"'
                + (' stroke="#24292f" stroke-width="2"' if host == target else "")
                + "/>"
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 6}" text-anchor="middle" '
                f'font-size="18" font-weight="700" fill="#ffffff">{symbol}</text></g>'
            )
    x = pad
    for state in legend:
        symbol, colour, text = STYLES[state]
        out.append(f'<rect x="{x}" y="{legend_y - 11}" width="14" height="14" rx="3" fill="{colour}"/>')
        out.append(f'<text x="{x + 19}" y="{legend_y}" fill="#24292f">{text}</text>')
        x += 25 + 7 * len(text)
    out.append(
        f'<text x="{pad}" y="{legend_y + 22}" fill="#57606a">bazel {html.escape(bazel)} · outlined: native '
        f'(build + test) · main @ {html.escape(commit[:7])}</text>'
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    jobs_path, out_dir, run_url, commit, bazel, stem = sys.argv[1:7]
    with open(jobs_path) as f:
        jobs = [json.loads(line) for line in f if line.strip()]
    cells = collect(jobs, bazel)
    with open(f"{out_dir}/{stem}.svg", "w") as f:
        f.write(render_svg(cells, commit, bazel))
    with open(f"{out_dir}/{stem}.json", "w") as f:
        json.dump(
            {
                "commit": commit,
                "run": run_url,
                "bazel": bazel,
                "platforms": PLATFORMS,
                "cells": [
                    {"host": h, "target": t, "kind": "native" if h == t else "cross",
                     "outcome": cells.get((h, t), "pending")}
                    for h in PLATFORMS
                    for t in PLATFORMS
                ],
            },
            f,
            indent=2,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
