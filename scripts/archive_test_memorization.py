"""
Memorization/Contamination Test for FOLIO and Multi-LogiEval

If the model memorized the benchmarks, it would:
1. Give original answer even when critical premises are removed
2. Not adjust to contradictory information added

A non-memorized model should:
1. Change answer when critical premise is removed (→ Unknown or flipped)
2. Change answer when contradiction is added
"""

import json
import asyncio
import random
from openai import AsyncOpenAI
import os
from datetime import datetime

# Test cases: (original, perturbed, expected_behavior)
# We'll create perturbations that SHOULD change the answer

FOLIO_PERTURBATION_TESTS = [
    {
        "id": "folio_test_1",
        "original_premises": [
            "All students who study hard pass the exam.",
            "John is a student.",
            "John studies hard."
        ],
        "original_conclusion": "John passes the exam.",
        "original_answer": "True",
        "perturbation_type": "remove_critical",
        "perturbed_premises": [
            "All students who study hard pass the exam.",
            "John is a student."
            # Removed: "John studies hard."
        ],
        "perturbed_conclusion": "John passes the exam.",
        "expected_perturbed_answer": "Unknown",
        "explanation": "Without knowing if John studies hard, we can't conclude he passes"
    },
    {
        "id": "folio_test_2",
        "original_premises": [
            "All mammals are warm-blooded.",
            "All whales are mammals.",
            "Moby is a whale."
        ],
        "original_conclusion": "Moby is warm-blooded.",
        "original_answer": "True",
        "perturbation_type": "add_contradiction",
        "perturbed_premises": [
            "All mammals are warm-blooded.",
            "All whales are mammals.",
            "Moby is a whale.",
            "Moby is not warm-blooded."  # Added contradiction
        ],
        "perturbed_conclusion": "Moby is warm-blooded.",
        "expected_perturbed_answer": "False",  # or should detect contradiction
        "explanation": "Direct contradiction added - model should recognize conflict"
    },
    {
        "id": "folio_test_3",
        "original_premises": [
            "If it rains, the ground gets wet.",
            "It is raining."
        ],
        "original_conclusion": "The ground is wet.",
        "original_answer": "True",
        "perturbation_type": "remove_critical",
        "perturbed_premises": [
            "If it rains, the ground gets wet."
            # Removed: "It is raining."
        ],
        "perturbed_conclusion": "The ground is wet.",
        "expected_perturbed_answer": "Unknown",
        "explanation": "Without knowing it's raining, we can't conclude ground is wet"
    },
    {
        "id": "folio_test_4",
        "original_premises": [
            "No reptiles have fur.",
            "All snakes are reptiles.",
            "Slither is a snake."
        ],
        "original_conclusion": "Slither has fur.",
        "original_answer": "False",
        "perturbation_type": "negate_rule",
        "perturbed_premises": [
            "All reptiles have fur.",  # Changed from "No reptiles have fur"
            "All snakes are reptiles.",
            "Slither is a snake."
        ],
        "perturbed_conclusion": "Slither has fur.",
        "expected_perturbed_answer": "True",
        "explanation": "Negating the rule should flip the answer"
    },
    {
        "id": "folio_test_5",
        "original_premises": [
            "Everyone who exercises regularly is healthy.",
            "Everyone who is healthy lives long.",
            "Alice exercises regularly."
        ],
        "original_conclusion": "Alice lives long.",
        "original_answer": "True",
        "perturbation_type": "break_chain",
        "perturbed_premises": [
            "Everyone who exercises regularly is healthy.",
            # Removed: "Everyone who is healthy lives long."
            "Alice exercises regularly."
        ],
        "perturbed_conclusion": "Alice lives long.",
        "expected_perturbed_answer": "Unknown",
        "explanation": "Breaking the inference chain should make conclusion unknown"
    },
]

MULTI_LOGI_PERTURBATION_TESTS = [
    {
        "id": "multi_test_1",
        "original_premises": [
            "All birds can fly.",
            "Tweety is a bird."
        ],
        "original_conclusion": "Tweety can fly.",
        "original_answer": "Yes",
        "perturbation_type": "add_exception",
        "perturbed_premises": [
            "All birds can fly.",
            "Tweety is a bird.",
            "Tweety is a penguin.",
            "Penguins cannot fly."
        ],
        "perturbed_conclusion": "Tweety can fly.",
        "expected_perturbed_answer": "No",
        "explanation": "Adding exception should change the answer"
    },
    {
        "id": "multi_test_2",
        "original_premises": [
            "If P then Q.",
            "If Q then R.",
            "P is true."
        ],
        "original_conclusion": "R is true.",
        "original_answer": "Yes",
        "perturbation_type": "remove_link",
        "perturbed_premises": [
            "If P then Q.",
            # Removed: "If Q then R."
            "P is true."
        ],
        "perturbed_conclusion": "R is true.",
        "expected_perturbed_answer": "No",  # or Unknown
        "explanation": "Removing chain link breaks the inference"
    },
]


async def test_single_question(client, premises, conclusion, model="gpt-5"):
    """Test a single question and return the answer."""

    premises_text = "\n".join(f"- {p}" for p in premises)

    prompt = f"""Given the following premises, determine if the conclusion is True, False, or Unknown.

Premises:
{premises_text}

Conclusion: {conclusion}

Think step by step, then provide your final answer as exactly one of: True, False, or Unknown."""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a logical reasoning assistant. Analyze the premises carefully and determine if the conclusion follows."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

        # Extract final answer
        answer_lower = answer.lower()
        if "unknown" in answer_lower.split()[-5:] or answer_lower.strip().endswith("unknown"):
            return "Unknown", answer
        elif "false" in answer_lower.split()[-5:] or answer_lower.strip().endswith("false") or "no" in answer_lower.split()[-3:]:
            return "False", answer
        elif "true" in answer_lower.split()[-5:] or answer_lower.strip().endswith("true") or "yes" in answer_lower.split()[-3:]:
            return "True", answer
        else:
            return "Unclear", answer

    except Exception as e:
        return f"Error: {e}", ""


async def run_memorization_test(model="gpt-5"):
    """Run the memorization test suite."""

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    all_tests = FOLIO_PERTURBATION_TESTS + MULTI_LOGI_PERTURBATION_TESTS

    results = []

    print("=" * 70)
    print("MEMORIZATION / CONTAMINATION TEST")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Testing {len(all_tests)} perturbation scenarios")
    print("=" * 70)

    for test in all_tests:
        print(f"\n--- Test: {test['id']} ({test['perturbation_type']}) ---")

        # Test original
        print("Testing ORIGINAL...")
        orig_answer, orig_response = await test_single_question(
            client,
            test['original_premises'],
            test['original_conclusion'],
            model
        )

        # Test perturbed
        print("Testing PERTURBED...")
        pert_answer, pert_response = await test_single_question(
            client,
            test['perturbed_premises'],
            test['perturbed_conclusion'],
            model
        )

        # Analyze
        answer_changed = orig_answer != pert_answer
        expected_change = test['original_answer'] != test['expected_perturbed_answer']

        if expected_change and not answer_changed:
            verdict = "POSSIBLE MEMORIZATION"
            symbol = "!!!"
        elif expected_change and answer_changed:
            if pert_answer == test['expected_perturbed_answer']:
                verdict = "CORRECT (answer changed appropriately)"
                symbol = "OK"
            else:
                verdict = f"CHANGED but unexpected (got {pert_answer}, expected {test['expected_perturbed_answer']})"
                symbol = "??"
        else:
            verdict = "N/A (no change expected)"
            symbol = "--"

        result = {
            "id": test['id'],
            "perturbation_type": test['perturbation_type'],
            "original_answer_expected": test['original_answer'],
            "original_answer_got": orig_answer,
            "perturbed_answer_expected": test['expected_perturbed_answer'],
            "perturbed_answer_got": pert_answer,
            "answer_changed": answer_changed,
            "verdict": verdict,
            "explanation": test['explanation']
        }
        results.append(result)

        print(f"  Original: expected={test['original_answer']}, got={orig_answer}")
        print(f"  Perturbed: expected={test['expected_perturbed_answer']}, got={pert_answer}")
        print(f"  [{symbol}] {verdict}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    possible_memorization = sum(1 for r in results if "MEMORIZATION" in r['verdict'])
    correct_changes = sum(1 for r in results if "CORRECT" in r['verdict'])

    print(f"Possible memorization cases: {possible_memorization}/{len(results)}")
    print(f"Correct behavior: {correct_changes}/{len(results)}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/memorization_test_{timestamp}.json"
    os.makedirs("results", exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            "model": model,
            "timestamp": timestamp,
            "summary": {
                "total_tests": len(results),
                "possible_memorization": possible_memorization,
                "correct_behavior": correct_changes
            },
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5", help="Model to test")
    args = parser.parse_args()

    asyncio.run(run_memorization_test(args.model))
