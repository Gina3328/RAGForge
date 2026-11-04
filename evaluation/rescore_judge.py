"""Experiment 3: re-score EXISTING RAGAS reports with a different judge LLM.

Nothing is retrieved or generated again. Each existing
evaluation/reports/ragas_report_<doc>_<arm>[_post_flush_fix].json already holds, per question, the
user_input, retrieved_contexts, response and reference. This script feeds exactly those to RAGAS
again, with another judge model, and writes a report in the same format into
evaluation/reports/rescored/<provider>_<model>/.

It must run in the RAGAS venv (it needs ragas and langchain-openai, which is already installed there):

    evaluation/.ragas_venv/bin/python evaluation/rescore_judge.py --provider openai --judge-model <name> --limit 3
    evaluation/.ragas_venv/bin/python evaluation/rescore_judge.py --provider openai --judge-model <name>
    evaluation/.ragas_venv/bin/python evaluation/rescore_judge.py --provider anthropic --judge-model <name>

Keys: OPENAI_API_KEY / ANTHROPIC_API_KEY from the environment or the project's .env.
Ollama must be running (answer_relevancy embeds text with the same embedding model as before).
It never touches Milvus, so it is safe to run while other experiments are running.

Resumable: a report that already exists in the output folder is skipped (use --force to redo it).
Design and decision rules: evaluation/ANALYSIS_PLAN_judges.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "evaluation" / "reports"

DOCS = ["ccby", "cdc", "ragbench"]
MAIN_ARMS = ["baseline", "generation_v2", "self_rag", "crag", "adaptive", "agentic"]
MERGE_ARMS = ["agentic_merge_rrf", "agentic_merge_dense"]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
RAGAS_COLUMNS = ("user_input", "retrieved_contexts", "response", "reference")


def default_inputs() -> list[str]:
    names = []
    for doc in DOCS:
        for arm in MAIN_ARMS:
            names.append(f"ragas_report_{doc}_{arm}_post_flush_fix.json")
        for arm in MERGE_ARMS:
            names.append(f"ragas_report_{doc}_{arm}.json")
    return names


def resolve_inputs(patterns: list[str] | None) -> list[Path]:
    if not patterns:
        paths = [REPORTS / n for n in default_inputs()]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            sys.exit("Missing input reports: " + ", ".join(missing))
        return paths
    out: list[Path] = []
    for pat in patterns:
        hits = sorted(glob.glob(str(REPORTS / pat)))
        if not hits:
            sys.exit(f"No report matches {pat!r} in {REPORTS}")
        out.extend(Path(h) for h in hits)
    return out


def make_llm(provider: str, model: str, temperature: float | None, max_tokens: int,
             reasoning_effort: str | None = None, token_param: str = "max_completion_tokens"):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": model, "max_tokens": max_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatAnthropic(**kwargs)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        # GPT-6 models are reasoning models: they take max_completion_tokens (reasoning tokens count against it)
        # and reject temperature unless reasoning effort is "none".
        kwargs = {"model": model, token_param: max_tokens}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning_effort not in (None, "none"):
            # RAGAS passes temperature=1e-8 on every call, which a reasoning model rejects with a 400 error.
            # Drop the field from the request itself, whatever RAGAS asks for.
            class ChatOpenAINoTemperature(ChatOpenAI):
                def _get_request_payload(self, input_, *, stop=None, **kw):
                    payload = super()._get_request_payload(input_, stop=stop, **kw)
                    payload.pop("temperature", None)
                    return payload

            return ChatOpenAINoTemperature(**kwargs)
        return ChatOpenAI(**kwargs)
    raise SystemExit(f"Unknown provider {provider!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-score existing RAGAS reports with another judge model.")
    ap.add_argument("--provider", required=True, choices=["openai", "anthropic"])
    ap.add_argument("--judge-model", required=True, help="Model name exactly as the provider's API expects it.")
    ap.add_argument(
        "--inputs",
        nargs="*",
        help="Report file names or glob patterns inside evaluation/reports/. Default: the 24 main reports "
        "(6 arms x 3 docs post_flush_fix, plus agentic_merge_rrf / agentic_merge_dense).",
    )
    ap.add_argument("--embedding-model", default="nomic-embed-text")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--no-temperature",
        action="store_true",
        help="Do not send a temperature (some models only accept their default).",
    )
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Default 4096; 8192 when a reasoning effort other than 'none' is used (reasoning tokens count).")
    ap.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], default=None,
                    help="OpenAI GPT-6 only. 'low' for gpt-6.1-sol; 'none' for gpt-6-luna. When the effort is not "
                    "'none' the temperature is not sent (the API rejects it).")
    ap.add_argument("--openai-token-param", choices=["max_completion_tokens", "max_tokens"], default="max_completion_tokens")
    ap.add_argument("--max-workers", type=int, default=8, help="Parallel judge calls inside RAGAS.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N questions of each report (smoke test).")
    ap.add_argument("--force", action="store_true", help="Redo reports that already exist in the output folder.")
    ap.add_argument("--dry-run", action="store_true", help="Load the inputs and build datasets, call no LLM.")
    ap.add_argument("--out-dir", default=None, help="Default: evaluation/reports/rescored/<provider>_<model>[_limitN]")
    args = ap.parse_args()

    inputs = resolve_inputs(args.inputs)
    reasoning = args.reasoning_effort if args.provider == "openai" else None
    max_tokens = args.max_tokens or (8192 if reasoning not in (None, "none") else 4096)
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{args.provider}_{args.judge_model}")
    if reasoning:
        tag += f"_re-{reasoning}"
    if args.limit:
        tag += f"_limit{args.limit}"
    out_dir = Path(args.out_dir) if args.out_dir else REPORTS / "rescored" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs first so that --dry-run needs no ragas / API at all.
    jobs = []
    for p in inputs:
        data = json.loads(p.read_text(encoding="utf-8"))
        samples = data["per_sample"]
        if args.limit:
            samples = samples[: args.limit]
        items = []
        for s in samples:
            ctx = s.get("retrieved_contexts")
            if isinstance(ctx, str):
                ctx = json.loads(ctx)
            items.append(
                {
                    "user_input": s["user_input"],
                    "retrieved_contexts": list(ctx or []),
                    "response": s["response"],
                    "reference": s["reference"],
                }
            )
        jobs.append((p, items))
    n_total = sum(len(i) for _, i in jobs)
    print(f"{len(jobs)} reports, {n_total} questions, output: {out_dir}")
    if args.dry_run:
        for p, items in jobs:
            print(f"  {p.name}: {len(items)} questions, "
                  f"{sum(1 for i in items if not i['retrieved_contexts'])} with empty contexts")
        return

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    key = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key):
        sys.exit(f"{key} not set (environment or project .env).")

    from langchain_community.embeddings import OllamaEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    temperature = None if args.no_temperature else args.temperature
    if reasoning not in (None, "none") and temperature is not None:
        print(f"Note: reasoning effort '{reasoning}' -> temperature is not sent (the API rejects it).")
        temperature = None
    chat = make_llm(args.provider, args.judge_model, temperature, max_tokens, reasoning, args.openai_token_param)

    # Fail fast on a wrong model name / key / unsupported parameter, before spending anything.
    try:
        reply = chat.invoke("Reply with the single word OK.")
        print("Smoke call ok:", str(getattr(reply, "content", reply))[:40])
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Judge model smoke call failed: {type(exc).__name__}: {exc}\n"
                 "(If it complains about temperature or max_tokens, try --no-temperature.)")

    judge_llm = LangchainLLMWrapper(chat)
    judge_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=args.embedding_model))
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    t0 = time.time()
    for k, (p, items) in enumerate(jobs, 1):
        out_path = out_dir / p.name
        if out_path.exists() and not args.force:
            print(f"[{k}/{len(jobs)}] skip (exists): {p.name}")
            continue
        print(f"[{k}/{len(jobs)}] {p.name}: {len(items)} questions ...", flush=True)
        dataset = EvaluationDataset.from_list(items)
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_emb,
            run_config=RunConfig(max_workers=args.max_workers, timeout=240),
        )
        df = result.to_pandas()
        avg = df[[m for m in METRIC_NAMES if m in df.columns]].mean(numeric_only=True).to_dict()
        nan_counts = {m: int(df[m].isna().sum()) for m in METRIC_NAMES if m in df.columns}
        # RAGAS swallows API errors and returns NaN. Do not save (and do not keep spending) if many scores failed.
        worst = max(nan_counts.values()) if nan_counts else 0
        if worst > 0.2 * len(df):
            print(f"\nSTOP: {worst}/{len(df)} scores failed (NaN) in {p.name}: {nan_counts}. "
                  "Not saved. Look at the 'Exception raised' lines above, fix the cause, and re-run "
                  "(finished reports are skipped).")
            sys.exit(1)
        report = {
            "averaged_scores": avg,
            "per_sample": df.to_dict(orient="records"),
            "judge": {
                "provider": args.provider,
                "model": args.judge_model,
                "temperature": temperature,
                "reasoning_effort": reasoning,
                "max_tokens": max_tokens,
                "embedding_model": args.embedding_model,
                "source_report": p.name,
                "nan_counts": nan_counts,
                "limit": args.limit,
            },
        }
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("    " + "  ".join(f"{m}={avg.get(m, float('nan')):.3f}" for m in METRIC_NAMES)
              + f"   NaN: {nan_counts}   elapsed {int(time.time() - t0)}s", flush=True)

    print(f"\nDone. Reports are in {out_dir}")
    print("Next: python3 evaluation/compare_judges.py --judges " + tag)


if __name__ == "__main__":
    main()
