from store import get_last_run_id, get_run_results

def detect_regressions(current_run_id: str, evaluator: str):
    previous_run_id = get_last_run_id()

    if not previous_run_id:
        print("No previous run found. Skipping regression check.")
        return

    current = get_run_results(current_run_id, evaluator)
    previous = get_run_results(previous_run_id, evaluator)

    regressions = []
    improvements = []

    for prompt, current_result in current.items():
        if prompt not in previous:
            continue
        prev = previous[prompt]
        curr = current_result

        if prev["passed"] and not curr["passed"]:
            regressions.append({
                "prompt": prompt,
                "prev_score": prev["score"],
                "curr_score": curr["score"]
            })
        elif not prev["passed"] and curr["passed"]:
            improvements.append({
                "prompt": prompt,
                "prev_score": prev["score"],
                "curr_score": curr["score"]
            })

    print(f"\n--- Regression report ({evaluator}) ---")

    if not regressions:
        print("No regressions detected.")
    else:
        print(f"{len(regressions)} regression(s) found:")
        for r in regressions:
            print(f"  REGRESSED: {r['prompt'][:60]}")
            print(f"    score: {r['prev_score']} -> {r['curr_score']}")

    if improvements:
        print(f"\n{len(improvements)} improvement(s):")
        for i in improvements:
            print(f"  IMPROVED: {i['prompt'][:60]}")
            print(f"    score: {i['prev_score']} -> {i['curr_score']}")