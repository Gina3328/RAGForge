"""Phase 2 of the decision-trace evaluation add-on: turn the JSON files
collect_decision_traces.py produces into the actual metrics and charts the
paper's core section needs (CRAG calibration curve, Adaptive routing
correctness, Agentic round efficiency, Self-RAG retrieve-or-not accuracy,
plus before/after deltas when two labeled runs of the same strategy are
given).

Runs in its OWN throwaway virtual environment, same isolation pattern as
run_ragas_eval.py: it only reads the plain-data decision_traces_*.json
files collect_decision_traces.py wrote, never imports `ragforge`, so its
dependencies (matplotlib, plus the raw `anthropic` package for the
correctness judge below) can never conflict with the main pipeline's
environment. Safe to delete the whole evaluation/ folder at any time.

--- Correctness judging ---

A decision trace only has the raw answer text and the reference answer --
nothing here knows yet whether the answer was actually *correct*. This
script asks the LLM (a single, deliberately minimal prompt) to judge each
answer against its reference and return CORRECT / INCORRECT / ABSTAINED.
ABSTAINED is its own outcome (not folded into INCORRECT) because for the
deliberately-unanswerable questions in each eval set, declining IS the
correct behavior -- conflating "wrongly declined an answerable question"
with "correctly declined an unanswerable one" would corrupt exactly the
abstention-quality signal the paper's Agentic/CRAG sections need. Reuses
the same ANTHROPIC_API_KEY already in .env -- no new account, no new key.

Usage (one-time setup, then run):

    python3 -m venv evaluation/.analysis_venv
    evaluation/.analysis_venv/bin/pip install -r evaluation/requirements-analysis.txt

    # Single run:
    evaluation/.analysis_venv/bin/python evaluation/analyze_decision_traces.py \\
        --traces evaluation/reports/decision_traces_crag.json \\
        --out evaluation/reports/metrics_crag.json

    # Before/after comparison (e.g. agentic_rag_strategy.py pre-fix vs post-fix):
    evaluation/.analysis_venv/bin/python evaluation/analyze_decision_traces.py \\
        --traces evaluation/reports/decision_traces_agentic_pre_fix.json \\
        --before-after evaluation/reports/decision_traces_agentic_post_fix.json \\
        --out evaluation/reports/metrics_agentic_before_after.json

Every run also writes one or more .png charts next to --out (same stem,
"_<chart-name>.png" suffix), using matplotlib with no third-party styling
dependency so this stays easy to keep running years from now.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed, just writes PNG files
import matplotlib.pyplot as plt  # noqa: E402

try:
    import anthropic  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The `anthropic` package is required for correctness judging. "
        "Install it into evaluation/.analysis_venv via requirements-analysis.txt."
    ) from exc

JUDGE_PROMPT = """You are grading one RAG system's answer against a reference answer.

Question: {question}

Reference answer: {reference}

System's answer: {answer}

Judge the system's answer as exactly one of:
CORRECT -- the answer conveys the same information as the reference (wording can differ).
INCORRECT -- the answer is wrong, incomplete in a way that misses the reference's key point(s), or fabricates information not in the reference.
ABSTAINED -- the system declined to answer / said the information wasn't available, rather than giving a substantive answer.

Reply with exactly one word: CORRECT, INCORRECT, or ABSTAINED."""

UNANSWERABLE_JUDGE_PROMPT = """You are grading whether a RAG system correctly declined to answer a
question that has NO answer in its knowledge base.

Question: {question}

System's answer: {answer}

Judge the system's answer as exactly one of:
CORRECT_ABSTENTION -- the system declined to answer, or said it couldn't find the information, rather than making something up.
HALLUCINATED -- the system gave a confident, substantive answer anyway (which must be fabricated, since no real answer exists).

Reply with exactly one word: CORRECT_ABSTENTION or HALLUCINATED."""


def judge_answer(
    client: anthropic.Anthropic, judge_model: str, question: str, reference: str | None, answer: str, answerable: bool
) -> str:
    prompt = (
        JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
        if answerable
        else UNANSWERABLE_JUDGE_PROMPT.format(question=question, answer=answer)
    )
    response = client.messages.create(
        model=judge_model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = response.content[0].text.strip().upper()
    if answerable:
        for label in ("CORRECT", "INCORRECT", "ABSTAINED"):
            if label in verdict:
                return label
        return "INCORRECT"  # unparseable judge output treated conservatively
    for label in ("CORRECT_ABSTENTION", "HALLUCINATED"):
        if label in verdict:
            return label
    return "HALLUCINATED"


def load_traces(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def grade_all(client: anthropic.Anthropic, judge_model: str, data: dict) -> None:
    """Adds a "verdict" field to each record in place."""
    for i, record in enumerate(data["records"], start=1):
        verdict = judge_answer(
            client, judge_model, record["question"], record.get("reference"), record["answer"], record["answerable"]
        )
        record["verdict"] = verdict
        print(f"  [{i}/{len(data['records'])}] {verdict}")


# --- Per-strategy metrics + charts --------------------------------------


def analyze_crag(data: dict, out_stem: Path) -> dict:
    """Calibration curve: bucket questions by CRAG's computed confidence,
    compare each bucket's average confidence against its actual
    correctness rate. A well-calibrated signal is roughly monotonic; a
    miscalibrated one (e.g. RRF-style score compression) is flat or
    inverted -- same evaluation logic Magnitude Mirage / Bacellar use for
    raw-similarity and fusion scores respectively, applied here to CRAG's
    own confidence score.
    """
    records = data["records"]
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    bucket_stats = []
    for lo, hi in buckets:
        in_bucket = [
            r for r in records if r["trace"].get("confidence") is not None and lo <= r["trace"]["confidence"] < hi
        ]
        if not in_bucket:
            continue
        avg_confidence = sum(r["trace"]["confidence"] for r in in_bucket) / len(in_bucket)
        correct = sum(1 for r in in_bucket if r["verdict"] in ("CORRECT", "CORRECT_ABSTENTION"))
        bucket_stats.append(
            {
                "range": f"[{lo:.1f}, {hi:.1f})",
                "n": len(in_bucket),
                "avg_confidence": avg_confidence,
                "correctness_rate": correct / len(in_bucket),
            }
        )

    route_counts: dict[str, int] = {}
    for r in records:
        route = r["trace"].get("route", "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if bucket_stats:
        xs = [b["avg_confidence"] for b in bucket_stats]
        ys = [b["correctness_rate"] for b in bucket_stats]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="perfect calibration")
        ax.plot(xs, ys, marker="o", color="#2563eb", linewidth=2, label="CRAG confidence")
        for b, x, y in zip(bucket_stats, xs, ys):
            ax.annotate(f"n={b['n']}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Average computed confidence")
    ax.set_ylabel("Actual correctness rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("CRAG confidence calibration")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_stem}_crag_calibration.png", dpi=150)
    plt.close(fig)

    return {"bucket_stats": bucket_stats, "route_counts": route_counts}


def analyze_adaptive(data: dict, out_stem: Path) -> dict:
    """Per-complexity-tier correctness: was each tier's routing decision
    actually the right call for that question, i.e. did it produce a
    correct answer?
    """
    records = data["records"]
    tiers = ["SIMPLE", "MEDIUM", "COMPLEX"]
    tier_stats = []
    for tier in tiers:
        in_tier = [r for r in records if r["trace"].get("complexity") == tier]
        if not in_tier:
            continue
        correct = sum(1 for r in in_tier if r["verdict"] in ("CORRECT", "CORRECT_ABSTENTION"))
        tier_stats.append({"tier": tier, "n": len(in_tier), "correctness_rate": correct / len(in_tier)})

    fig, ax = plt.subplots(figsize=(5, 4))
    if tier_stats:
        ax.bar(
            [t["tier"] for t in tier_stats],
            [t["correctness_rate"] for t in tier_stats],
            color=["#22c55e", "#f59e0b", "#ef4444"][: len(tier_stats)],
        )
        for i, t in enumerate(tier_stats):
            ax.text(i, t["correctness_rate"] + 0.02, f"n={t['n']}", ha="center", fontsize=8)
    ax.set_ylabel("Correctness rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Adaptive-RAG: correctness by routed complexity tier")
    fig.tight_layout()
    fig.savefig(f"{out_stem}_adaptive_routing.png", dpi=150)
    plt.close(fig)

    return {"tier_stats": tier_stats}


def analyze_self_rag(data: dict) -> dict:
    records = data["records"]
    should_retrieve_true = sum(1 for r in records if r["trace"].get("should_retrieve") is True)
    avg_retries = sum(r["trace"].get("retries_used", 0) for r in records) / len(records) if records else 0.0
    correct = sum(1 for r in records if r["verdict"] in ("CORRECT", "CORRECT_ABSTENTION"))
    return {
        "n": len(records),
        "should_retrieve_rate": should_retrieve_true / len(records) if records else 0.0,
        "avg_retries_used": avg_retries,
        "correctness_rate": correct / len(records) if records else 0.0,
    }


def analyze_agentic(data: dict, out_stem: Path, label: str = "") -> dict:
    """Round-efficiency: how many rounds each question actually took,
    against whether the final answer was correct -- flags cases that
    burned every available round without needing to, or stopped too
    early and got it wrong.
    """
    records = data["records"]
    round_counts = [r["trace"]["num_rounds"] for r in records]
    correct_flags = [r["verdict"] in ("CORRECT", "CORRECT_ABSTENTION") for r in records]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#22c55e" if c else "#ef4444" for c in correct_flags]
    ax.bar(range(1, len(records) + 1), round_counts, color=colors)
    ax.set_xlabel("Question #")
    ax.set_ylabel("Rounds used")
    ax.set_title(f"Agentic RAG: rounds per question{f' ({label})' if label else ''}")
    green_patch = plt.Rectangle((0, 0), 1, 1, color="#22c55e", label="correct")
    red_patch = plt.Rectangle((0, 0), 1, 1, color="#ef4444", label="incorrect")
    ax.legend(handles=[green_patch, red_patch], fontsize=8)
    fig.tight_layout()
    suffix = f"_{label}" if label else ""
    fig.savefig(f"{out_stem}_agentic_rounds{suffix}.png", dpi=150)
    plt.close(fig)

    return {
        "n": len(records),
        "avg_rounds": sum(round_counts) / len(round_counts) if round_counts else 0.0,
        "max_rounds": max(round_counts) if round_counts else 0,
        "correctness_rate": sum(correct_flags) / len(correct_flags) if correct_flags else 0.0,
    }


ANALYZERS = {
    "crag": analyze_crag,
    "adaptive": analyze_adaptive,
    "self_rag": lambda data, out_stem: analyze_self_rag(data),
    "agentic": analyze_agentic,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade + analyze one or two decision-trace runs.")
    parser.add_argument("--traces", required=True, help="Path to a decision_traces_*.json file.")
    parser.add_argument(
        "--before-after",
        default=None,
        help="Optional second decision_traces_*.json file (same strategy, same questions) "
        "for a before/after comparison, e.g. pre-fix vs post-fix.",
    )
    parser.add_argument("--out", required=True, help="Path to write the metrics JSON to; charts are saved alongside it.")
    parser.add_argument(
        "--judge-model",
        default="claude-haiku-4-5",
        help="Anthropic model used to grade each answer against its reference. Defaults to the same "
        "model config/stage4.yml's llm.model uses -- pass a different one to match your own config.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set -- this script reads it directly from the environment (no .env loader in this isolated venv; export it or copy the value from the project's .env before running).")
    client = anthropic.Anthropic(api_key=api_key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_stem = out_path.with_suffix("")

    print(f"Grading {args.traces} ...")
    data = load_traces(args.traces)
    grade_all(client, args.judge_model, data)
    strategy = data["strategy"]

    analyzer = ANALYZERS[strategy]
    if strategy == "agentic":
        metrics = analyzer(data, out_stem, label="before" if args.before_after else "")
    else:
        metrics = analyzer(data, out_stem)

    result = {"strategy": strategy, "run": metrics}

    if args.before_after:
        print(f"Grading {args.before_after} ...")
        data_after = load_traces(args.before_after)
        grade_all(client, args.judge_model, data_after)
        if data_after["strategy"] != strategy:
            raise SystemExit("--before-after file must be the same strategy as --traces.")
        analyzer_after = ANALYZERS[strategy]
        metrics_after = (
            analyzer_after(data_after, out_stem, label="after") if strategy == "agentic" else analyzer_after(data_after, out_stem)
        )
        result["before"] = metrics
        result["after"] = metrics_after
        result.pop("run", None)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nWrote metrics to {out_path}")
    print(f"Chart(s) written alongside it as {out_stem}_*.png")


if __name__ == "__main__":
    main()
