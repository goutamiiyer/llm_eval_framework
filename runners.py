import json
import uuid
from evaluator import run_exact_match, run_llm_judge, run_semantic
from hallucination_eval import run_hallucination_evals
from store import init_db, save_results, get_last_runs
from regression import detect_regressions

import time

def load_test_cases(path: str = "test_cases.jsonl") -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_suite(evaluators: list) -> dict:
    test_cases = load_test_cases()
    run_id = str(uuid.uuid4())[:8]
    init_db()

    do_exact = "exact_match" in evaluators
    do_judge = "llm_judge" in evaluators
    do_semantic = "semantic" in evaluators
    do_hallucination = "hallucination" in evaluators

    results = []
    flat_results = []

    if do_exact or do_judge or do_semantic:
        print("\n--- Eval run ---\n")

        for case in test_cases:
            exact = run_exact_match(case["prompt"], case["expected"]) if do_exact else None
            judge = run_llm_judge(case["prompt"], case["expected"]) if do_judge else None
            semantic = run_semantic(
                case["prompt"],
                case.get("expected_semantic", case["expected"])
            ) if do_semantic else None

            time.sleep(1)

            results.append({
                "prompt": case["prompt"],
                "exact": exact,
                "judge": judge,
                "semantic": semantic
            })

            print(f"Prompt: {case['prompt'][:60]}")
            if exact:
                print(f"  Exact match : {'PASS' if exact['passed'] else 'FAIL'} (score: {exact['score']})")
            if judge:
                print(f"  LLM judge   : {'PASS' if judge['passed'] else 'FAIL'} (score: {judge['score']})")
            if semantic:
                print(f"  Semantic    : {'PASS' if semantic['passed'] else 'FAIL'} (score: {semantic['score']})")
            print()

        exact_score = judge_score = semantic_score = 0.0

        if do_exact:
            exact_score = sum(r["exact"]["score"] for r in results) / len(results)
            print(f"Overall exact match score : {exact_score:.0%}")
            for r in results:
                flat_results.append({
                    "prompt": r["prompt"],
                    "expected": r["exact"]["expected"],
                    "response": r["exact"]["response"],
                    "evaluator": "exact_match",
                    "score": r["exact"]["score"],
                    "passed": r["exact"]["passed"],
                    "reason": ""
                })

        if do_judge:
            judge_score = sum(r["judge"]["score"] for r in results) / len(results)
            print(f"Overall LLM judge score   : {judge_score:.0%}")
            for r in results:
                flat_results.append({
                    "prompt": r["prompt"],
                    "expected": r["judge"]["expected"],
                    "response": r["judge"]["response"],
                    "evaluator": "llm_judge",
                    "score": r["judge"]["score"],
                    "passed": r["judge"]["passed"],
                    "reason": r["judge"].get("reason", "")
                })

        if do_semantic:
            semantic_score = sum(r["semantic"]["score"] for r in results) / len(results)
            print(f"Overall semantic score    : {semantic_score:.2f}")
            for r in results:
                flat_results.append({
                    "prompt": r["prompt"],
                    "expected": r["semantic"]["expected"],
                    "response": r["semantic"]["response"],
                    "evaluator": "semantic",
                    "score": r["semantic"]["score"],
                    "passed": r["semantic"]["passed"],
                    "reason": r["semantic"].get("reason", "")
                })

        save_results(run_id, flat_results)
        print(f"\nRun saved with ID: {run_id}")

        if do_exact:
            detect_regressions(run_id, "exact_match")
        if do_judge:
            detect_regressions(run_id, "llm_judge")

    if do_hallucination:
        hallucination_results = run_hallucination_evals()
        flat_hall = []
        with open("hallucination_cases.jsonl") as f:
            hall_cases = [json.loads(line) for line in f]
        for i, c in enumerate(hall_cases):
            r = hallucination_results[i]
            flat_hall.append({
                "prompt": c["prompt"],
                "expected": "refuse",
                "response": r["response"],
                "evaluator": "hallucination",
                "score": r["score"],
                "passed": r["passed"],
                "reason": r["reason"]
            })
        save_results(run_id, flat_hall)

    print("\n--- Last 5 runs ---")
    for row in get_last_runs():
        print(f"Run {row[0]} | {row[1][:19]} | {row[2]} | avg score: {row[3]:.2f} | cases: {row[4]}")

    return {
        "run_id": run_id,
        "results": results
    }


def generate_markdown_report(results: list, run_id: str) -> str:
    lines = [
        "# LLM Eval Report\n",
        f"Run ID: {run_id}\n",
        "## Per-case results\n"
    ]
    for r in results:
        lines.append(f"### {r['prompt'][:80]}")
        if r.get("exact"):
            lines.append(f"- Exact match: {'PASS' if r['exact']['passed'] else 'FAIL'} ({r['exact']['score']})")
        if r.get("judge"):
            lines.append(f"- LLM judge: {'PASS' if r['judge']['passed'] else 'FAIL'} ({r['judge']['score']})")
            lines.append(f"- Judge reason: {r['judge'].get('reason', '')}")
        if r.get("semantic"):
            lines.append(f"- Semantic: {'PASS' if r['semantic']['passed'] else 'FAIL'} ({r['semantic']['score']})\n")
    return "\n".join(lines)