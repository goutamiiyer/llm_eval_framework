import sys
sys.path.insert(0, ".")
from evaluator import run_exact_match

result = run_exact_match("Who painted the Mona Lisa?", "Leonardo da Vinci")
print("Response:", result["response"])
print("Passed:", result["passed"])