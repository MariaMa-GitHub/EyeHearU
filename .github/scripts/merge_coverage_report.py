#!/usr/bin/env python3
"""Build README coverage section (and optional PR comment body) from coverage JSON files.

Used locally or in custom automation — the default GitHub Actions workflow does not run this script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


def _fmt_pct_display(pct: float) -> str:
    if abs(pct - round(pct)) < 0.005:
        return str(int(round(pct)))
    s = f"{pct:.2f}".rstrip("0").rstrip(".")
    return s


def _load_py_coverage(path: Path) -> tuple[float, str, float | None]:
    """Returns (line_pct, branch table cell markdown, branch_pct or None if N/A)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    t = data["totals"]
    line_pct = round(float(t["percent_covered"]), 2)
    nb = int(t.get("num_branches") or 0)
    cb = int(t.get("covered_branches") or 0)
    if nb > 0:
        branch_pct = round(100.0 * cb / nb, 2)
        branch_cell = f"**{_fmt_pct_display(branch_pct)}%**"
        return line_pct, branch_cell, branch_pct
    branch_cell = "—"
    return line_pct, branch_cell, None


def _load_jest_metrics(path: Path) -> tuple[float, float, float, float]:
    """Returns lines, branches, statements, functions (percent)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    total = data["total"]
    lines = round(float(total["lines"]["pct"]), 2)
    branches = round(float(total["branches"]["pct"]), 2)
    statements = round(float(total["statements"]["pct"]), 2)
    functions = round(float(total["functions"]["pct"]), 2)
    return lines, branches, statements, functions


def _shield_slug(pct: float) -> str:
    if pct >= 95:
        return "brightgreen"
    if pct >= 80:
        return "green"
    if pct >= 70:
        return "yellow"
    if pct >= 50:
        return "orange"
    return "red"


def _badge_url(display_name: str, pct: float) -> str:
    """shields.io badge URL; label and message are percent-encoded."""
    color = _shield_slug(pct)
    left = f"coverage: {display_name}"
    right = f"{_fmt_pct_display(pct)}%"
    return (
        f"https://img.shields.io/badge/"
        f"{quote(left, safe='')}-{quote(right, safe='')}-{color}"
    )


def _badge_markdown(display_name: str, pct: float) -> str:
    url = _badge_url(display_name, pct)
    return f"![{display_name} coverage]({url})"


def _py_badge_pct(line_pct: float, branch_pct: float | None) -> float:
    """Conservative badge % when both line and branch are measured."""
    if branch_pct is None:
        return line_pct
    return min(line_pct, branch_pct)


def build_markdown(
    *,
    backend_json: Path,
    ml_json: Path,
    jest_json: Path,
    run_url: str | None,
    for_pr: bool,
) -> str:
    b_line, b_branch_cell, b_branch_pct = _load_py_coverage(backend_json)
    m_line, m_branch_cell, m_branch_pct = _load_py_coverage(ml_json)
    j_lines, j_branches, j_stmts, j_funcs = _load_jest_metrics(jest_json)
    j_worst = min(j_lines, j_branches, j_stmts, j_funcs)

    b_badge = _py_badge_pct(b_line, b_branch_pct)
    m_badge = _py_badge_pct(m_line, m_branch_pct)

    lines = [
        _badge_markdown("Backend", b_badge)
        + " "
        + _badge_markdown("ML", m_badge)
        + " "
        + _badge_markdown("Mobile", j_worst),
        "",
        "| Component | Lines | Branches |",
        "|-----------|-------|----------|",
        f"| Backend | **{_fmt_pct_display(b_line)}%** | {b_branch_cell} |",
        f"| ML | **{_fmt_pct_display(m_line)}%** | {m_branch_cell} |",
        f"| Mobile | **{_fmt_pct_display(j_lines)}%** | **{_fmt_pct_display(j_branches)}%** |",
        "",
    ]
    if for_pr:
        lines.append(
            "_Enforced in CI: `pytest --cov-fail-under=100` (backend + ML, lines and branches) and Jest "
            "`coverageThreshold` at 100% (mobile)._"
        )
        if run_url:
            lines.extend(["", f"[View this workflow run]({run_url})"])
    else:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"<sub>Last CI update: {now}</sub>")
        if run_url:
            lines.append(f"<sub><a href=\"{run_url}\">Workflow run</a></sub>")

    return "\n".join(lines) + "\n"


def patch_readme(readme: Path, block: str) -> bool:
    text = readme.read_text(encoding="utf-8")
    pattern = (
        r"<!-- COVERAGE_TABLE_START -->.*?"
        r"<!-- COVERAGE_TABLE_END -->"
    )
    replacement = f"<!-- COVERAGE_TABLE_START -->\n{block}<!-- COVERAGE_TABLE_END -->"
    new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if n != 1:
        print("README: markers <!-- COVERAGE_TABLE_START/END --> not found", file=sys.stderr)
        return False
    if new_text == text:
        return False
    readme.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend-json", type=Path, required=True)
    p.add_argument("--ml-json", type=Path, required=True)
    p.add_argument("--jest-json", type=Path, required=True)
    p.add_argument("--run-url", default="", help="Link to the GitHub Actions run")
    p.add_argument("--pr-comment", type=Path, help="Write PR comment body to this file")
    p.add_argument("--readme", type=Path, help="Patch README.md between coverage markers")
    p.add_argument("--for-pr", action="store_true", help="PR wording (no HTML sub)")
    args = p.parse_args()

    for path in (args.backend_json, args.ml_json, args.jest_json):
        if not path.is_file():
            print(f"Missing coverage file: {path}", file=sys.stderr)
            return 1

    run_url = args.run_url.strip() or None
    body = build_markdown(
        backend_json=args.backend_json,
        ml_json=args.ml_json,
        jest_json=args.jest_json,
        run_url=run_url,
        for_pr=args.for_pr,
    )

    if args.pr_comment:
        args.pr_comment.write_text(body, encoding="utf-8")

    if args.readme:
        # Do not treat "no change" as an error; reserve non-zero exits for
        # clearly invalid situations (e.g., missing coverage files above).
        patch_readme(args.readme, body)

    if not args.pr_comment and not args.readme:
        sys.stdout.write(body)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
