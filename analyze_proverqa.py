import json

with open("results/proverqa/hard/gpt-5_20251216_225946/all_results.json") as f:
    results = json.load(f)

api_errors = 0
wrong_answer = 0
unknown_pred = 0
total_wrong = 0

for r in results:
    if not r["correct"]:
        total_wrong += 1
        if r["prediction"] == "UNKNOWN":
            unknown_pred += 1
            if "ERROR" in str(r.get("model_response", "")):
                api_errors += 1
        else:
            wrong_answer += 1

print(f"Total questions: 500")
print(f"Correct: {500 - total_wrong}")
print(f"Total wrong: {total_wrong}")
print()
print("Error Breakdown:")
print(f"  1. API errors (no response): {api_errors}")
print(f"  2. Wrong A/B/C answers: {wrong_answer}")
print(f"  3. Parsing failures (UNKNOWN but has response): {unknown_pred - api_errors}")
