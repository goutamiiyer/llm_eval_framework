from groq import Groq
import os
from dotenv import load_dotenv

import json

from store import init_db, save_results, get_last_runs
import uuid

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def evaluate_exact_match(response: str, expected: str) -> dict:
    response_clean = response.strip().lower()
    expected_clean = expected.strip().lower()
    passed = expected_clean in response_clean
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "response": response,
        "expected": expected
    }

def run_eval(prompt: str, expected: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return evaluate_exact_match(answer, expected)

def evaluate_with_llm_judge(prompt: str, response: str, expected: str) -> dict:
    judge_prompt = f"""You are an expert evaluator grading an LLM response.

Question: {prompt}
Expected answer: {expected}
Actual response: {response}

Grade the response on a scale of 0 to 1:
- 1.0 means fully correct and complete
- 0.5 means partially correct
- 0.0 means wrong or missing the point

Respond in this exact format:
SCORE: <number>
REASON: <one sentence>"""

    judgment = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": judge_prompt}]
    )
    
    text = judgment.choices[0].message.content.strip()
    
    score_line = [l for l in text.split("\n") if l.startswith("SCORE:")][0]
    reason_line = [l for l in text.split("\n") if l.startswith("REASON:")][0]
    
    score = float(score_line.replace("SCORE:", "").strip())
    reason = reason_line.replace("REASON:", "").strip()
    
    return {
        "passed": score >= 0.5,
        "score": score,
        "response": response,
        "expected": expected,
        "reason": reason
    }


def run_eval_with_judge(prompt: str, expected: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return evaluate_with_llm_judge(prompt, answer, expected)


with open("test_cases.jsonl") as f:
    test_cases = [json.loads(line) for line in f]

print("\n--- Full eval run: exact match vs LLM-as-judge ---\n")

results = []
for case in test_cases:
    exact = run_eval(case["prompt"], case["expected"])
    judge = run_eval_with_judge(case["prompt"], case["expected"])
    
    results.append({"exact": exact, "judge": judge, "prompt": case["prompt"]})
    
    print(f"Prompt: {case['prompt'][:60]}")
    print(f"  Exact match : {'PASS' if exact['passed'] else 'FAIL'} (score: {exact['score']})")
    print(f"  LLM judge   : {'PASS' if judge['passed'] else 'FAIL'} (score: {judge['score']})")
    print(f"  Judge reason: {judge['reason']}")
    print()

exact_score = sum(r["exact"]["score"] for r in results) / len(results)
judge_score = sum(r["judge"]["score"] for r in results) / len(results)

print(f"Overall exact match score : {exact_score:.0%}")
print(f"Overall LLM judge score   : {judge_score:.0%}")

# save to SQLite
run_id = str(uuid.uuid4())[:8]
init_db()

flat_results = []
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
    flat_results.append({
        "prompt": r["prompt"],
        "expected": r["judge"]["expected"],
        "response": r["judge"]["response"],
        "evaluator": "llm_judge",
        "score": r["judge"]["score"],
        "passed": r["judge"]["passed"],
        "reason": r["judge"].get("reason", "")
    })

save_results(run_id, flat_results)
print(f"\nRun saved with ID: {run_id}")

print("\n--- Last 5 runs ---")
for row in get_last_runs():
    print(f"Run {row[0]} | {row[1][:19]} | {row[2]} | avg score: {row[3]:.2f} | cases: {row[4]}")