# LLM Eval Framework

A lightweight framework for evaluating LLM outputs automatically.
Built from scratch in Python. No bloated dependencies, no magic.

## What it does

Takes a list of test cases, sends each prompt to an LLM, checks the
response against the expected answer using two different evaluators,
and saves every run to a local database so you can track scores over time.

Built with Groq (free tier) and Llama 3.1, but designed so the LLM
provider is swappable without touching the evaluator logic.

## Why this matters

Most developers just call an LLM and hope it works. Evaluation is how
you actually know if it works, across different inputs, consistently,
across runs. This project catches regressions when you swap models,
change prompts, or add new features to an AI system.

## Project structure

```
LLM_eval_framework/
├── evaluator.py       # core eval logic, API calls, main loop
├── store.py           # SQLite persistence, run history
├── test_cases.jsonl   # test dataset: prompts + expected answers
└── .env               # API keys, never committed to git
```

## How to run

1. Clone the repo
2. Install dependencies
```
pip install groq python-dotenv
```
3. Create a .env file with your Groq API key
```
GROQ_API_KEY=your_key_here
```
4. Run the evaluator
```
python evaluator.py
```

## Sample output

```
--- Full eval run: exact match vs LLM-as-judge ---

Prompt: What is the capital of Australia?
  Exact match : PASS (score: 1.0)
  LLM judge   : PASS (score: 0.8)
  Judge reason: Correctly identifies Canberra but lacks sentence framing.

Prompt: List exactly 2 colors. No more, no less. No explanation.
  Exact match : FAIL (score: 0.0)
  LLM judge   : PASS (score: 1.0)
  Judge reason: Response listed exactly 2 colors as requested.

Overall exact match score : 93%
Overall LLM judge score   : 89%

Run saved with ID: e551fef4

--- Last 5 runs ---
Run e551fef4 | 2026-06-22T20:18:04 | exact_match | avg score: 0.93 | cases: 15
Run e551fef4 | 2026-06-22T20:18:04 | llm_judge   | avg score: 0.89 | cases: 15
```

## Evaluator types

**Exact match** checks if the expected answer appears in the response.
Fast, deterministic, cheap. Works well for factual questions with fixed
answers. Breaks down for open-ended responses or strict instruction
following tasks.

**LLM-as-judge** sends the prompt, response, and expected answer to a
second LLM call and asks it to grade quality on a 0 to 1 scale with a
reason. Handles nuance that exact match cannot.

## Key finding

Exact match scored 93%, LLM judge scored 89% on the same 15 test cases.
They fail differently.

Exact match misses instruction following violations because it only
checks for keyword presence. A response that adds extra words when told
not to will still pass exact match if the keyword is there.

LLM judge sometimes penalizes correct answers for lacking context it
was never asked to provide. This is called criteria drift, a known
limitation of LLM-as-judge evaluation.

Neither evaluator is sufficient alone. The right approach is using both
and understanding what each one catches.

### Key finding: semantic similarity needs sentence-level expected answers

Initial semantic scoring returned 0.51 average despite most answers
being factually correct. Root cause: expected answers were single
words or numbers like "100" or "Canberra", while LLM responses were
full sentences. Embedding models compare meaning at the sentence level,
so comparing "The boiling point of water is 100 degrees Celsius" against
"100" produces near-zero similarity even when the answer is right.

Fix: added `expected_semantic` field to test cases with full sentence
versions of expected answers. Semantic evaluator uses this field when
available, falls back to `expected` otherwise.

Lesson: choose your evaluator based on the shape of your data, not
just the type of question. Short factual lookups belong with exact
match. Semantic similarity earns its place when responses and expected
answers are both rich, paragraph-length text.

## Run history

Every eval run is saved to a local SQLite database with a unique run ID,
timestamp, per-case scores, and judge reasoning. This makes it possible
to track score changes across runs and catch regressions when prompts
or models change.

## Regression detection

Every run is compared against the previous run automatically. If a
test case that was passing before starts failing, it gets flagged
immediately with the score drop shown.

Example output after a prompt change:

```
--- Regression report (exact_match) ---
1 regression(s) found:
  REGRESSED: What is 2 + 2?
    score: 1.0 -> 0.0
```

This is what makes the framework useful in practice. You can change
a prompt, swap a model, or add new test cases, and know within one
run whether anything broke.

## What's coming next

- Semantic similarity scoring using embeddings
- GitHub Actions CI to run evals automatically on every commit
- Hallucination-specific test cases

## Tech stack

- Python 3.13
- Groq API (free tier)
- Llama 3.1 8B Instant
- SQLite (built into Python, zero setup)
- python-dotenv

## Author

Goutami, software engineer with a background in distributed systems
and backend infrastructure at Microsoft Azure, building toward applied
AI engineering roles. This project is part of a public portfolio
documenting that transition.