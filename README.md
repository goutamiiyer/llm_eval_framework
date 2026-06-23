# LLM Eval Framework

A lightweight framework for evaluating LLM outputs automatically.
Built from scratch in Python. No bloated dependencies, no magic.

## What it does

Takes a list of test cases, sends each prompt to an LLM, checks the
response against the expected answer, and reports a pass/fail score.

Built with Groq (free tier) and Llama 3.1, but designed so the LLM
provider is swappable without touching the evaluator logic.

## Why this matters

Most developers just call an LLM and hope it works. Evaluation is how
you actually know if it works, across different inputs, reliably.
This project is the foundation for catching regressions when you
swap models, change prompts, or add new features to an AI system.

## Project structure

```
LLM_eval_framework/
├── evaluator.py       # core eval logic and API call
├── test_cases.jsonl   # test dataset: prompts + expected answers
└── .env               # API keys, never committed to git
```

## How to run

1. Clone the repo
2. Install dependencies
   pip install groq python-dotenv
3. Create a .env file with your Groq API key
   GROQ_API_KEY=your_key_here
4. Run the evaluator
   python evaluator.py

## Sample output

PASS: What is the capital of Australia?
PASS: What is 2 + 2?
FAIL: What is 15 multiplied by 13?

Overall score: 90%

## Evaluator types

**Exact match** — checks if the expected answer appears in the response.
Fast, deterministic, works well for factual questions with fixed answers.
Breaks down for open-ended responses.

**LLM-as-judge** — sends the prompt, response, and expected answer to
a second LLM call and asks it to grade quality on a 0 to 1 scale with
a reason. Handles nuance that exact match can't.

Known limitation: LLM-as-judge has a self-similarity bias where the
judge tends to favor responses that sound like its own output style.
For production use, the judge model should differ from the model being
evaluated.

## Example LLM-as-judge output

```
Score: 1.0
Reason: The response accurately explains Rayleigh scattering as the
reason the sky appears blue, fully matching the expected answer.
Response: The sky appears blue because of Rayleigh scattering, where
shorter blue wavelengths scatter more efficiently through atmospheric
gas molecules.
```

## What's coming next

- Semantic similarity scoring using embeddings
- SQLite score history to track results over time
- GitHub Actions CI to run evals automatically on every commit

## Tech stack

- Python 3.13
- Groq API (free tier)
- Llama 3.1 8B Instant
- python-dotenv

## Author

Goutami — software engineer transitioning into applied AI engineering.
Background in distributed systems and backend infrastructure at Microsoft Azure.
Building this as part of a portfolio for AI engineering roles.