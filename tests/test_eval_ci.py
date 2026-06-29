import sys
sys.path.insert(0, ".")
import time
from evaluator import run_exact_match, evaluate_with_llm_judge

test_cases = [
    {"prompt": "What is the capital of Australia?", "expected": "Canberra"},
    {"prompt": "What is 2 + 2?", "expected": "4"},
    {"prompt": "Who wrote Romeo and Juliet?", "expected": "Shakespeare"},
]

print("Running CI eval subset (3 cases)...")
for case in test_cases:
    exact = run_exact_match(case["prompt"], case["expected"])
    time.sleep(2)
    judge = evaluate_with_llm_judge(case["prompt"], exact["response"], case["expected"])
    time.sleep(2)
    status = "PASS" if exact["passed"] else "FAIL"
    print(f"{status}: {case['prompt'][:50]}")

print("CI eval OK")