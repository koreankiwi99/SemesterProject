#!/usr/bin/env python3
"""Check result files for errors (API errors, Lean errors, connection errors)."""

import json
import os
from pathlib import Path

BASE_DIR = Path("/Users/kyuheekim/SemesterProject")

# Main result files used in the report
RESULT_FILES = [
    # GPT-5 FOLIO
    "results/folio/cot/all_results.json",
    "results/folio/lean/all_results.json",
    "results/folio/bidirectional/all_results.json",
    "results/folio/two_stage/all_results.json",
    # GPT-5 Multi-LogiEval all depths
    "results/multilogieval/all_depths/cot/all_results.json",
    "results/multilogieval/all_depths/lean/all_results.json",
    "results/multilogieval/all_depths/bidirectional/all_results.json",
    "results/multilogieval/all_depths/two_stage/all_results.json",
    # GPT-5 Multi-LogiEval d5 only
    "results/multilogieval/d5_only/cot/all_results.json",
    "results/multilogieval/d5_only/lean/all_results.json",
    "results/multilogieval/d5_only/bidirectional/all_results.json",
    # Other models FOLIO
    "results/cot/folio/deepseek_r1/all_results.json",
    "results/cot/folio/mistral_large/all_results.json",
    "results/cot/folio/qwen3_235b/all_results.json",
    # Other models Multi-LogiEval
    "results/cot/multilogieval/deepseek_r1/all_results.json",
    "results/cot/multilogieval/mistral_large/all_results.json",
    "results/cot/multilogieval/qwen3_235b/all_results.json",
    # GPT-5 Simple/Minimal reasoning
    "results/archive/folio_simple_20251205_220912/all_results.json",
    "results/archive/multilogieval_simple_20251205_221137/all_results.json",
    "results/archive/folio_cot_minimal_reasoning_20251205_221312/all_results.json",
    "results/archive/multilogieval_cot_minimal_reasoning_20251205_221348/all_results.json",
    "results/archive/folio_simple_minimal_reasoning_20251206_115637/all_results.json",
    "results/archive/multilogieval_simple_minimal_reasoning_20251206_115713/all_results.json",
]


def check_file(filepath):
    """Check a result file for errors."""
    full_path = BASE_DIR / filepath

    if not full_path.exists():
        return {"status": "FILE_NOT_FOUND", "path": filepath}

    try:
        with open(full_path) as f:
            data = json.load(f)
    except Exception as e:
        return {"status": "JSON_ERROR", "path": filepath, "error": str(e)}

    if not isinstance(data, list):
        return {"status": "INVALID_FORMAT", "path": filepath}

    total = len(data)
    errors = []
    lean_errors = 0
    api_errors = 0
    connection_errors = 0

    for i, item in enumerate(data):
        # Check for explicit error field
        if "error" in item:
            err = item["error"]
            errors.append({"index": i, "error": err[:100] if isinstance(err, str) else str(err)[:100]})

            if isinstance(err, str):
                if "connection" in err.lower() or "timeout" in err.lower():
                    connection_errors += 1
                elif "api" in err.lower() or "rate" in err.lower():
                    api_errors += 1

        # Check for Lean verification errors (in lean/bidirectional results)
        if "lean_verification" in item:
            lv = item["lean_verification"]
            if isinstance(lv, dict) and not lv.get("success", True):
                lean_errors += 1

        # Check nested results (FOLIO CoT format)
        if "results" in item and isinstance(item["results"], list):
            for j, sub in enumerate(item["results"]):
                if "error" in sub:
                    errors.append({"index": f"{i}.{j}", "error": str(sub["error"])[:100]})

    return {
        "status": "OK" if not errors else "HAS_ERRORS",
        "path": filepath,
        "total_records": total,
        "error_count": len(errors),
        "lean_errors": lean_errors,
        "api_errors": api_errors,
        "connection_errors": connection_errors,
        "sample_errors": errors[:5] if errors else []
    }


def main():
    print("=" * 80)
    print("ERROR CHECK FOR RESULT FILES USED IN SUPERVISOR REPORT")
    print("=" * 80)

    all_ok = True

    for filepath in RESULT_FILES:
        result = check_file(filepath)

        short_path = filepath.replace("results/", "").replace("/all_results.json", "")

        if result["status"] == "FILE_NOT_FOUND":
            print(f"❌ {short_path}: FILE NOT FOUND")
            all_ok = False
        elif result["status"] == "JSON_ERROR":
            print(f"❌ {short_path}: JSON ERROR - {result['error']}")
            all_ok = False
        elif result["status"] == "HAS_ERRORS":
            print(f"⚠️  {short_path}: {result['error_count']} errors out of {result['total_records']} records")
            if result.get("lean_errors"):
                print(f"    - Lean errors: {result['lean_errors']}")
            if result.get("api_errors"):
                print(f"    - API errors: {result['api_errors']}")
            if result.get("connection_errors"):
                print(f"    - Connection errors: {result['connection_errors']}")
            if result.get("sample_errors"):
                for e in result["sample_errors"][:3]:
                    print(f"    - Sample: [{e['index']}] {e['error']}")
            all_ok = False
        else:
            print(f"✓  {short_path}: OK ({result['total_records']} records)")

    print("=" * 80)
    if all_ok:
        print("✓ All files OK - no errors found")
    else:
        print("⚠️  Some files have issues - review above")


if __name__ == "__main__":
    main()
