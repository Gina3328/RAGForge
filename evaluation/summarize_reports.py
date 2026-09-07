"""Combine the per-variant RAGAS reports from run_ragas_eval.py into one
side-by-side comparison table.

Reads evaluation/reports/ragas_report_<variant>.json for each variant in
collect_predictions.VARIANTS (falling back to the un-suffixed
evaluation/reports/ragas_report.json for "baseline", since that was
already the default output filename before --variant existed -- so an
earlier baseline run doesn't need to be redone just to show up here), and
prints a table with each metric plus its row-over-row delta: each row
shows what turning on one more feature bought you over the row before it.

Runs in the normal .venv (no RAGAS/ragas_venv needed -- this only reads
plain JSON already written by run_ragas_eval.py):

    uv run python evaluation/summarize_reports.py

Skips (and says so) any variant whose report file doesn't exist yet, so
this can be run after collecting only some of the variants and re-run
again later as more come in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_predictions import VARIANTS  # noqa: E402 (reused for labels + row order)

REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"

METRICS = ["faithfulness", "answer_relevancy", "hit_rate", "context_precision", "context_recall"]


def _report_path(variant_name: str) -> Path:
    """Where to look for one variant's scored report.

    Every variant's normal location is ragas_report_<variant>.json (what
    the loop in evaluation/README.md writes). "baseline" alone also
    accepts the older unsuffixed ragas_report.json, since that's the
    filename run_ragas_eval.py's --out default has always used.
    """
    suffixed = REPORTS_DIR / f"ragas_report_{variant_name}.json"
    if suffixed.exists():
        return suffixed
    if variant_name == "baseline":
        return REPORTS_DIR / "ragas_report.json"
    return suffixed  # caller checks .exists() and reports it as missing


def main() -> None:
    rows: list[tuple[str, dict]] = []
    for variant_name in VARIANTS:
        path = _report_path(variant_name)
        if not path.exists():
            print(f"(skipping {variant_name!r} -- {path.name} not found yet)")
            continue
        with path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        rows.append((variant_name, report.get("averaged_scores", {})))

    if not rows:
        print(
            "No reports found yet. Run collect_predictions.py, then "
            "run_ragas_eval.py, for at least one --variant first."
        )
        return

    header = ["variant"] + METRICS
    col_widths = [max(len(h), 12) for h in header]
    for variant_name, _ in rows:
        col_widths[0] = max(col_widths[0], len(variant_name))

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(cell.ljust(w) for cell, w in zip(cells, col_widths))

    print(fmt_row(header))
    print("-+-".join("-" * w for w in col_widths))

    # Tracks each metric's last-seen value across rows (not necessarily
    # the immediately previous row) so a metric missing from one row (e.g.
    # hit_rate before reference_chunk_ids are filled in) doesn't break the
    # delta for the next row that does have it.
    previous: dict[str, float] = {}
    for variant_name, scores in rows:
        cells = [variant_name]
        for metric in METRICS:
            value = scores.get(metric)
            if value is None:
                cells.append("-")
                continue
            if metric in previous:
                delta = value - previous[metric]
                sign = "+" if delta >= 0 else ""
                cells.append(f"{value:.3f} ({sign}{delta:.3f})")
            else:
                cells.append(f"{value:.3f}")
            previous[metric] = value
        print(fmt_row(cells))

    print(f"\n({len(rows)}/{len(VARIANTS)} variants have a report so far)")


if __name__ == "__main__":
    main()
