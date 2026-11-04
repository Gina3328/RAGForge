"""Experiment 7, step 2: label every collected answer as REFUSAL or ANSWER with an LLM judge.

Reads evaluation/reports/refusal_<doc>_<arm>.json (written by run_refusal.sh; either a decision-trace file or a
predictions file) and the question files refusal_mixed_<doc>.json. For every (document, arm, question) it asks the
judge only one thing: did the response refuse / say the information is not available, or did it give a specific
answer to the question? It does NOT judge whether the answer is correct.

The fixed CRAG decline message ("Sorry, I couldn't find enough relevant information ...") is labelled REFUSAL by a
rule, without calling the judge.

Run in the RAGAS venv (it has langchain-anthropic and langchain-openai):

    evaluation/.ragas_venv/bin/python evaluation/classify_refusals.py --provider anthropic --judge-model <name> --limit 5
    evaluation/.ragas_venv/bin/python evaluation/classify_refusals.py --provider anthropic --judge-model <name>
    evaluation/.ragas_venv/bin/python evaluation/classify_refusals.py --provider openai --judge-model <name>   # second judge

Keys: ANTHROPIC_API_KEY / OPENAI_API_KEY from the environment or the project's .env. Does not touch Milvus.
Output: evaluation/reports/refusal_labels_<provider>_<model>.json (resumable) and refusal_spotcheck_<tag>.csv
(40 answers for you to check by hand).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "evaluation" / "reports"
DOCS = ["ccby", "cdc", "ragbench"]
ARMS = ["baseline", "generation_v2", "self_rag", "crag", "adaptive", "agentic_rrf", "agentic_dense"]
CANNED = "sorry, i couldn't find enough relevant information"

PROMPT = """You are labelling how a question-answering system responded to a question about a document.

Question: {q}

System response:
\"\"\"{a}\"\"\"

Label the response with exactly one of two labels.
REFUSAL: the response says the information is not available, not found, or not stated in the documents or context, or it otherwise declines to answer the question that was asked. A refusal that then adds related background still counts as REFUSAL as long as it does not give a specific answer to the question that was asked.
ANSWER: the response gives a specific answer to the question that was asked, correct or not, even if it hedges.
Do not judge whether the answer is correct.

Reply with JSON only: {{"label": "REFUSAL" or "ANSWER", "reason": "<one short sentence>"}}"""


def load_items(arms: list[str]) -> list[dict]:
    items = []
    for doc in DOCS:
        mixed_path = REPORTS / f"refusal_mixed_{doc}.json"
        if not mixed_path.exists():
            continue
        meta = {q["question"]: q for q in json.loads(mixed_path.read_text(encoding="utf-8"))}
        for arm in arms:
            p = REPORTS / f"refusal_{doc}_{arm}.json"
            if not p.exists():
                continue
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                pairs = [(r["question"], r["answer"]) for r in raw["records"]]
            else:
                pairs = [(r["user_input"], r["response"]) for r in raw]
            for q, a in pairs:
                m = meta.get(q)
                if m is None:
                    print(f"WARNING: question not in {mixed_path.name}: {q[:60]!r}", file=sys.stderr)
                    continue
                items.append({"key": f"{doc}|{arm}|{q}", "doc": doc, "arm": arm, "question": q, "answer": a or "",
                              "set": m.get("set"), "qtype": m.get("type")})
    return items


def make_llm(provider: str, model: str, temperature: float | None, reasoning_effort: str | None = None):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kw = {"model": model, "max_tokens": 300}
        if temperature is not None:
            kw["temperature"] = temperature
        return ChatAnthropic(**kw)
    from langchain_openai import ChatOpenAI

    # GPT-6 models are reasoning models: max_completion_tokens (reasoning tokens count against it), and
    # temperature is rejected unless the reasoning effort is "none".
    kw = {"model": model, "max_completion_tokens": 2048 if reasoning_effort not in (None, "none") else 300}
    if reasoning_effort:
        kw["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        kw["temperature"] = temperature
    return ChatOpenAI(**kw)


def parse(text: str):
    m = re.search(r'"label"\s*:\s*"(REFUSAL|ANSWER)"', text)
    if not m:
        return None, text[:120]
    r = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
    return m.group(1), (r.group(1) if r else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai"])
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-temperature", action="store_true")
    ap.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], default=None,
                    help="OpenAI GPT-6 only ('low' for gpt-6.1-sol, 'none' for gpt-6-luna). Temperature is not sent "
                    "when the effort is not 'none'.")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N answers (smoke test; writes a _limitN file).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = load_items(args.arms)
    if not items:
        sys.exit("No refusal_<doc>_<arm>.json files found; run evaluation/run_refusal.sh first.")
    if args.limit:
        items = items[: args.limit]
    reasoning = args.reasoning_effort if args.provider == "openai" else None
    tag = (re.sub(r"[^A-Za-z0-9._-]+", "-", f"{args.provider}_{args.judge_model}")
           + (f"_re-{reasoning}" if reasoning else "") + (f"_limit{args.limit}" if args.limit else ""))
    out_path = REPORTS / f"refusal_labels_{tag}.json"
    n_rule = sum(1 for i in items if i["answer"].strip().lower().startswith(CANNED))
    print(f"{len(items)} answers ({n_rule} labelled by the CRAG-decline rule), output: {out_path.name}")
    if args.dry_run:
        return

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    key = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key):
        sys.exit(f"{key} not set (environment or project .env).")
    temp = None if (args.no_temperature or reasoning not in (None, "none")) else args.temperature
    chat = make_llm(args.provider, args.judge_model, temp, reasoning)
    try:
        chat.invoke("Reply with the single word OK.")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Judge smoke call failed: {type(exc).__name__}: {exc}\n(If it complains about temperature, use --no-temperature.)")

    labels: dict = {}
    if out_path.exists():
        labels = json.loads(out_path.read_text(encoding="utf-8")).get("labels", {})
    todo = []
    for it in items:
        if it["key"] in labels and labels[it["key"]].get("label") in ("REFUSAL", "ANSWER"):
            continue
        if it["answer"].strip().lower().startswith(CANNED):
            labels[it["key"]] = {"label": "REFUSAL", "reason": "fixed CRAG decline message", "by": "rule"}
        else:
            todo.append(it)
    print(f"{len(todo)} answers to label with the judge ({len(items) - len(todo)} already done or rule-labelled)")

    def work(it):
        prompt = PROMPT.format(q=it["question"], a=it["answer"][:4000])
        last = ""
        for _ in range(3):
            try:
                resp = chat.invoke(prompt)
                label, reason = parse(str(getattr(resp, "content", resp)))
                if label:
                    return it["key"], {"label": label, "reason": reason, "by": "llm"}
                last = reason
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"[:200]
        return it["key"], {"label": "ERROR", "reason": last, "by": "llm"}

    def save():
        out_path.write_text(json.dumps({
            "judge": {"provider": args.provider, "model": args.judge_model, "temperature": args.temperature},
            "labels": labels}, ensure_ascii=False, indent=1), encoding="utf-8")

    done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for f in as_completed(futs):
            k, v = f.result()
            labels[k] = v
            done += 1
            if done % 50 == 0:
                save()
                print(f"  {done}/{len(todo)}", flush=True)
    save()
    errors = sum(1 for v in labels.values() if v["label"] == "ERROR")
    print(f"Done. {len(labels)} labels, {errors} errors. Written {out_path}")

    # Spot-check sheet: 20 REFUSAL + 20 ANSWER, random, fixed seed. The person fills in column 'your_label'.
    by_key = {i["key"]: i for i in items}
    rng = random.Random(0)
    pick = []
    for lab in ("REFUSAL", "ANSWER"):
        pool = sorted(k for k, v in labels.items() if v["label"] == lab and k in by_key)
        pick += rng.sample(pool, min(20, len(pool)))
    sheet = REPORTS / f"refusal_spotcheck_{tag}.csv"
    with sheet.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "question", "response_first_700_chars", "judge_label", "your_label", "note"])
        for k in pick:
            it = by_key[k]
            w.writerow([k, it["question"], it["answer"][:700].replace("\n", " "), labels[k]["label"], "", ""])
    print(f"Spot-check sheet: {sheet}")


if __name__ == "__main__":
    main()
