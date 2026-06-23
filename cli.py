import argparse
from store import get_last_runs
from runners import run_suite, generate_markdown_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Eval Framework: evaluate LLM outputs across multiple evaluator types"
    )
    parser.add_argument(
        "--evaluator",
        choices=["exact_match", "llm_judge", "semantic", "hallucination", "all"],
        default="all",
        help="Which evaluator to run (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to a markdown file (e.g. --output report.md)"
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print run history from the database and exit"
    )
    return parser.parse_args()


def print_history():
    rows = get_last_runs(10)
    print("\n--- Run history (last 10) ---\n")
    for row in rows:
        print(f"Run {row[0]} | {row[1][:19]} | {row[2]:<15} | avg score: {row[3]:.2f} | cases: {row[4]}")


if __name__ == "__main__":
    args = parse_args()

    if args.history:
        print_history()
        exit(0)

    evaluators = (
        ["exact_match", "llm_judge", "semantic", "hallucination"]
        if args.evaluator == "all"
        else [args.evaluator]
    )

    output = run_suite(evaluators)

    if args.output:
        report = generate_markdown_report(output["results"], output["run_id"])
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")