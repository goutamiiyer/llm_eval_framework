from groq import Groq
import os
from dotenv import load_dotenv
import json

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

result = run_eval("What is the capital of Australia?", "Canberra")
print(result)

with open("test_cases.jsonl") as f:
    test_cases = [json.loads(line) for line in f]

results = []
for case in test_cases:
    result = run_eval(case["prompt"], case["expected"])
    results.append(result)
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {case['prompt']}")

score = sum(r["score"] for r in results) / len(results)
print(f"\nOverall score: {score:.0%}")
