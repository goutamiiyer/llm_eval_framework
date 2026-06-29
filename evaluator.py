from groq import Groq
from semantic_eval import evaluate_semantic_similarity
import os
from dotenv import load_dotenv

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
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": judge_prompt}]
    )

    text = judgment.choices[0].message.content.strip()

    try:
        score_line = [l for l in text.split("\n") if l.startswith("SCORE:")][0]
        reason_line = [l for l in text.split("\n") if l.startswith("REASON:")][0]
        score = float(score_line.replace("SCORE:", "").strip())
        reason = reason_line.replace("REASON:", "").strip()
    except (IndexError, ValueError):
        score = 0.0
        reason = "Could not parse judge response"

    return {
        "passed": score >= 0.5,
        "score": score,
        "response": response,
        "expected": expected,
        "reason": reason
    }


def run_exact_match(prompt: str, expected: str) -> dict:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return evaluate_exact_match(answer, expected)


def run_llm_judge(prompt: str, expected: str) -> dict:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return evaluate_with_llm_judge(prompt, answer, expected)


def run_semantic(prompt: str, expected: str) -> dict:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return evaluate_semantic_similarity(answer, expected)