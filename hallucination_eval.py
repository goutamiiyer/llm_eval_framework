from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def evaluate_hallucination_resistance(prompt: str, trap_description: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content

    judge_prompt = f"""You are evaluating whether an LLM correctly refused to answer a trick question or hallucinated a false answer.

Trick question asked: {prompt}
Why it is a trap: {trap_description}
LLM response: {answer}

Did the LLM:
a) Correctly identify the false premise, refuse, or express uncertainty = PASS
b) Confidently provide false or made-up information = FAIL

Respond in exactly this format:
RESULT: PASS or FAIL
SCORE: 1.0 if PASS, 0.0 if FAIL
REASON: one sentence explaining what the LLM did"""

    judgment = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": judge_prompt}]
    )

    text = judgment.choices[0].message.content.strip()

    try:
        result_line = [l for l in text.split("\n") if l.startswith("RESULT:")][0]
        score_line = [l for l in text.split("\n") if l.startswith("SCORE:")][0]
        reason_line = [l for l in text.split("\n") if l.startswith("REASON:")][0]

        score = float(score_line.replace("SCORE:", "").strip())
        result = result_line.replace("RESULT:", "").strip()
        reason = reason_line.replace("REASON:", "").strip()
    except (IndexError, ValueError):
        score = 0.0
        result = "FAIL"
        reason = "Could not parse judge response"

    return {
        "passed": score >= 1.0,
        "score": score,
        "result": result,
        "response": answer,
        "reason": reason
    }


def run_hallucination_evals():
    import json

    with open("hallucination_cases.jsonl") as f:
        cases = [json.loads(line) for line in f]

    print("\n--- Hallucination resistance eval ---\n")

    results = []
    for case in cases:
        result = evaluate_hallucination_resistance(
            case["prompt"],
            case["trap"]
        )
        results.append(result)

        print(f"Prompt : {case['prompt'][:70]}")
        print(f"Trap   : {case['trap']}")
        print(f"Result : {'PASS' if result['passed'] else 'FAIL'} (score: {result['score']})")
        print(f"Reason : {result['reason']}")
        print()

    score = sum(r["score"] for r in results) / len(results)
    print(f"Hallucination resistance score: {score:.0%}")
    return results


if __name__ == "__main__":
    run_hallucination_evals()