"""
Batch runner to generate and validate all Multi-LogiEval combinations.
"""

import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data_generation"))
from generate_multilogi_samples import generate_samples, COMBINATION_RULES

def run_all_combinations(depth: str = "d7", temperature: float = 0.7,
                        model: str = "gpt-4o", validate_logic: bool = True):
    """Generate samples for all 5 combinations, parse, and validate them."""

    results_dir = Path("results/multilogi/generated")
    report_dir = Path("results/multilogi/reports")
    final_dir = Path(f"data/multi_logi_extended/{depth}_Data/pl")

    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"{'='*70}")
    print("STEP 1: GENERATING SAMPLES")
    print(f"{'='*70}\n")

    for combination in range(1, 6):
        rule_name = COMBINATION_RULES[combination]
        print(f"Combination {combination} ({rule_name})...")

        raw_file = results_dir / f"combination_{combination}_{depth}_raw.txt"

        try:
            response_text = generate_samples(
                combination=combination,
                depth=depth,
                temperature=temperature,
                model=model
            )

            with open(raw_file, 'w') as f:
                f.write(response_text)
            print(f"  Saved to: {raw_file}\n")

        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({
                "combination": combination,
                "rule": rule_name,
                "status": "generation_failed",
                "error": str(e)
            })

    print(f"\n{'='*70}")
    print("STEP 2: VALIDATING SAMPLES")
    print(f"{'='*70}\n")

    for combination in range(1, 6):
        rule_name = COMBINATION_RULES[combination]
        raw_file = results_dir / f"combination_{combination}_{depth}_raw.txt"
        temp_json = results_dir / f"combination_{combination}_{depth}_temp.json"
        report_file = report_dir / f"combination_{combination}_{depth}_report.txt"

        if not raw_file.exists():
            continue

        print(f"Combination {combination} ({rule_name})...")

        try:
            cmd = [
                "python", "src/data_generation/parse_and_validate.py",
                "--raw-file", str(raw_file),
                "--combination", str(combination),
                "--depth", depth,
                "--output-json", str(temp_json),
                "--report-file", str(report_file)
            ]

            if validate_logic:
                cmd.append("--validate-logic")

            subprocess.run(cmd, check=True)

            results.append({
                "combination": combination,
                "rule": rule_name,
                "status": "validated",
                "files": {
                    "raw": str(raw_file),
                    "temp_json": str(temp_json),
                    "report": str(report_file)
                }
            })

        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({
                "combination": combination,
                "rule": rule_name,
                "status": "validation_failed",
                "error": str(e)
            })

    print(f"\n{'='*70}")
    print("STEP 3: MOVING VALIDATED DATA")
    print(f"{'='*70}\n")

    final_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        if result["status"] != "validated":
            continue

        combination = result["combination"]
        rule_name = result["rule"]
        temp_json = Path(result["files"]["temp_json"])
        final_file = final_dir / f"{rule_name}.json"

        try:
            import shutil
            shutil.copy(temp_json, final_file)
            result["files"]["final_json"] = str(final_file)
            result["status"] = "success"
            print(f"Combination {combination}: {final_file}")

        except Exception as e:
            print(f"Combination {combination}: ERROR - {e}")
            result["status"] = "move_failed"
            result["error"] = str(e)

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}\n")

    for result in results:
        if result["status"] == "success":
            print(f"Combination {result['combination']} ({result['rule']}): SUCCESS")
            print(f"  Raw: {result['files']['raw']}")
            print(f"  Report: {result['files']['report']}")
            print(f"  Final JSON: {result['files']['final_json']}\n")
        else:
            print(f"Combination {result['combination']} ({result['rule']}): {result['status'].upper()}")
            if 'error' in result:
                print(f"  Error: {result['error']}\n")

    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"{'='*70}")
    print(f"Completed: {success_count}/5 combinations")
    print(f"Results in: results/multilogi/")
    print(f"Final data in: {final_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=str, default="d7")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--no-validate-logic", action="store_true",
                       help="Skip GPT-4o logical validation")

    args = parser.parse_args()

    run_all_combinations(
        depth=args.depth,
        temperature=args.temperature,
        model=args.model,
        validate_logic=not args.no_validate_logic
    )
