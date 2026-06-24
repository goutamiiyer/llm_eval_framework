# LLM Eval Framework

A lightweight framework for evaluating LLM outputs automatically.
Built from scratch in Python. No bloated dependencies, no magic.

## What it does

Takes a list of test cases, sends each prompt to an LLM, scores each
response using three different evaluators, saves every run to a local
SQLite database, and automatically flags regressions between runs.

Built with Groq (free tier) and Llama 3.1, but the LLM provider is
swappable without touching any evaluator logic.

## Why this matters

Most developers call an LLM and hope it works. Evaluation is how you
actually know if it works, consistently, across inputs, across runs.
This framework catches regressions when you swap models, change prompts,
or add new features to an AI system.

## Project structure

```
llm_eval_framework/
├── evaluator.py            # core eval logic, API calls, main loop
├── semantic_eval.py        # sentence embedding similarity evaluator
├── regression.py           # detects score drops between runs
├── store.py                # SQLite persistence and run history
├── test_cases.jsonl        # test dataset with exact and semantic expected answers
├── .github/
│   └── workflows/
│       └── eval_ci.yml     # runs evals automatically on every push
└── .env                    # API keys, never committed to git
```

## How to run

1. Clone the repo
2. Install dependencies
```
pip install groq python-dotenv sentence-transformers
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
--- Full eval run: exact match vs LLM-as-judge vs Semantic ---

Prompt: What is the capital of Australia?
  Exact match : PASS (score: 1.0)
  LLM judge   : PASS (score: 1.0)
  Semantic    : PASS (score: 0.8731)

Prompt: List exactly 2 colors. No more, no less. No explanation.
  Exact match : FAIL (score: 0.0)
  LLM judge   : PASS (score: 1.0)
  Semantic    : FAIL (score: 0.4893)

Overall exact match score : 93%
Overall LLM judge score   : 93%
Overall semantic score    : 0.86

Run saved with ID: 4906ed02

--- Regression report (exact_match) ---
No regressions detected.
```

## Evaluator types

**Exact match** checks if the expected answer appears in the response.
Fast, deterministic, zero API cost. Works well for factual questions
with fixed answers. Misses instruction following violations because it
only checks for keyword presence.

**LLM-as-judge** sends the prompt, response, and expected answer to a
second LLM call and asks it to grade quality 0 to 1 with a reason.
Handles nuance exact match cannot. Known limitation: criteria drift,
where the judge penalizes correct answers for lacking context it was
never asked to provide.

**Semantic similarity** converts both the response and expected answer
into embedding vectors using a local sentence-transformers model and
measures cosine similarity. Deterministic and free, no API call needed.
Works best when both response and expected answer are full sentences.

## Key findings

**Exact match vs LLM judge fail differently.** Exact match misses
instruction following violations. LLM judge sometimes invents grading
criteria. Neither is sufficient alone.

**Semantic similarity breaks on short expected answers.** Initial
scoring returned 0.51 average despite correct answers. Root cause:
comparing a full LLM sentence against a single word like "100" or
"Canberra" produces near-zero cosine similarity even when factually
correct. Fix: added `expected_semantic` field with full sentence
versions of expected answers. Score jumped to 0.86 after the fix.

Lesson: pick your evaluator based on the shape of your data, not just
the question type.

**Hallucination resistance**

A separate eval suite tests whether the LLM correctly refuses to answer
trick questions rather than confidently making things up.

*Test categories:*
- Fictional sources (non-existent studies, researchers, papers)
- Impossible quotes (attributing statements to people who predated the topic)
- Fictional events (awards, earthquakes, sports tournaments that never happened)
- False premises (questions that embed a wrong assumption)

Latest result: 88% resistance score across 8 hallucination probes.

*Real failure caught:*

```
Prompt: Who won the 2021 FIFA World Cup?
Trap: There was no World Cup in 2021, they happen every 4 years
Result: FAIL
Reason: LLM confidently fabricated a Copa America story with
incorrect details rather than flagging the false premise
```

This is what makes hallucination testing valuable. The model didn't
hedge or express uncertainty. It invented specific, plausible-sounding
false information. Exact match and semantic similarity would never
catch this class of failure.

**Non-determinism finding**

Running the same 8 hallucination probes three times in one session
produced scores of 100%, 88%, and 38%. Same model, same prompts,
different outputs each time.

This is not a framework bug. It demonstrates why single-run evals
are unreliable for LLM systems. The SQLite run history exists
specifically to surface this variance over time.

Production implication: any LLM-powered feature that needs consistent
behavior requires either temperature=0, output validation, or
multiple-run averaging, not a one-time eval pass.

## Regression detection

Every run is compared against the previous run automatically. Cases
that drop from PASS to FAIL are flagged with the score shown.

```
--- Regression report (exact_match) ---
1 regression(s) found:
  REGRESSED: What is 2 + 2?
    score: 1.0 -> 0.0
```

Improvements are tracked too. You always know whether a change helped
or hurt.

## CI pipeline

GitHub Actions runs the full eval suite on every push to main. Results
are visible in the Actions tab. The sentence-transformers model is
cached between runs so it doesn't re-download every time.

## CLI usage

Run specific evaluators without touching the code:

```bash
python evaluator.py                          # runs everything
python evaluator.py --evaluator exact_match  # only exact match
python evaluator.py --evaluator llm_judge    # only LLM judge
python evaluator.py --evaluator semantic     # only semantic
python evaluator.py --evaluator hallucination # only hallucination suite
python evaluator.py --evaluator all --output report.md  # save markdown report
python evaluator.py --history                # print run history and exit
```

## Tech stack

- Python 3.11
- Groq API (free tier)
- Llama 3.1 8B Instant
- sentence-transformers (all-MiniLM-L6-v2, runs locally)
- SQLite (built into Python, zero setup)
- GitHub Actions for CI

## Author

Goutami, software engineer with a background in distributed systems
and backend infrastructure at Microsoft Azure, building toward applied
AI engineering roles. This project is part of a public portfolio
documenting that transition.