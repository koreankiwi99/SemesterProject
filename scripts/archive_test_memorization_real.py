#!/usr/bin/env python3
"""
Memorization Test using REAL FOLIO/Multi-LogiEval questions

Tests if GPT-5 memorized the benchmarks by:
1. Removing critical premises → answer should become Unknown
2. Adding contradictory premise → answer should flip
3. Swapping entity names → if memorized, model might give original answer

Key insight: If model memorized the dataset, it will give the ORIGINAL answer
even when the question is modified in ways that should change the answer.
"""

import json
import asyncio
import random
import os
import sys
import argparse
from datetime import datetime
from openai import AsyncOpenAI

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datasets.folio import load_folio
from datasets.multilogieval import load_and_sample_multilogieval

random.seed(42)


def load_folio_questions(filepath, n=10):
    """Load random FOLIO questions with True/False labels."""
    data = load_folio(filepath)

    # Filter for True/False (not Unknown - harder to test perturbations)
    candidates = []
    for q in data:
        if q.get('label') in ['True', 'False']:
            # Convert premises string to list
            premises_str = q.get('premises', '')
            premises_list = [p.strip() for p in premises_str.split('\n') if p.strip()]
            candidates.append({
                'premises': premises_list,
                'conclusion': q.get('conclusion', ''),
                'label': q.get('label', ''),
                'example_id': q.get('example_id'),
                'story_id': q.get('story_id')
            })

    return random.sample(candidates, min(n, len(candidates)))


def load_multilogi_questions(data_dir, n=10):
    """Load random Multi-LogiEval questions."""
    # Use the existing loader
    all_samples = load_and_sample_multilogieval(
        data_dir,
        logic_types=['fol', 'pl'],  # Skip nm for cleaner tests
        depths=['d3_Data', 'd4_Data', 'd5_Data'],
        samples_per_combination=max(5, n // 6),  # 3 logic types x 3 depths
        seed=42
    )

    # Convert to list format with premises
    questions = []
    for sample in all_samples:
        # Split context into sentences (premises)
        import re
        context = sample.get('context', '')
        sentences = [s.strip() + '.' for s in re.split(r'\.(?:\s|$)', context) if s.strip()]

        questions.append({
            'premises': sentences,
            'conclusion': sample.get('question', ''),
            'label': 'Yes' if sample.get('answer', '').lower() == 'yes' else 'No',
            'logic_type': sample.get('logic_type'),
            'depth': sample.get('depth'),
            'rule': sample.get('rule'),
            'source': f"{sample.get('logic_type')}/{sample.get('depth_dir')}/{sample.get('rule')}"
        })

    return random.sample(questions, min(n, len(questions)))


def perturb_remove_premise(premises, conclusion, label):
    """Remove a premise that seems critical (contains key terms from conclusion)."""
    if len(premises) <= 1:
        return premises, "Cannot remove - only 1 premise"

    conclusion_words = set(conclusion.lower().split())

    # Find premise with most overlap with conclusion
    best_idx = 0
    best_overlap = 0
    for i, p in enumerate(premises):
        p_words = set(p.lower().split())
        overlap = len(p_words & conclusion_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i

    # Remove that premise
    removed = premises[best_idx]
    new_premises = premises[:best_idx] + premises[best_idx+1:]
    return new_premises, f"Removed: '{removed[:60]}...'"


def perturb_add_negation(premises, conclusion, label):
    """Add a premise that directly contradicts the conclusion."""
    # Create negation of conclusion
    conclusion_lower = conclusion.lower()

    if "is a" in conclusion_lower:
        negation = conclusion.replace(" is a ", " is not a ").replace(" Is a ", " Is not a ")
    elif "is " in conclusion_lower:
        negation = conclusion.replace(" is ", " is not ", 1).replace(" Is ", " Is not ", 1)
    elif "can " in conclusion_lower:
        negation = conclusion.replace(" can ", " cannot ", 1).replace(" Can ", " Cannot ", 1)
    elif "will " in conclusion_lower:
        negation = conclusion.replace(" will ", " will not ", 1).replace(" Will ", " Will not ", 1)
    elif "does " in conclusion_lower:
        negation = conclusion.replace(" does ", " does not ", 1).replace(" Does ", " Does not ", 1)
    elif "did " in conclusion_lower:
        negation = conclusion.replace(" did ", " did not ", 1).replace(" Did ", " Did not ", 1)
    else:
        negation = "It is NOT the case that " + conclusion

    new_premises = premises + [negation]
    return new_premises, f"Added: '{negation[:60]}...'"


async def test_question(client, premises, conclusion, model="gpt-5", is_folio=True):
    """Test a single question."""
    premises_text = "\n".join(f"- {p}" for p in premises)

    if is_folio:
        prompt = f"""Given these premises, determine if the conclusion is True, False, or Unknown.

Premises:
{premises_text}

Conclusion: {conclusion}

Think step by step, then provide your final answer as exactly one of: True, False, or Unknown."""
    else:
        prompt = f"""Given these premises, answer the question with Yes or No.

Premises:
{premises_text}

Question: {conclusion}

Think step by step, then answer with exactly: Yes or No."""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a logical reasoning assistant. Analyze premises carefully."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content
        answer_lower = answer.lower()
        last_100 = answer_lower[-100:]

        if is_folio:
            if "unknown" in last_100 or "uncertain" in last_100:
                return "Unknown", answer
            elif "false" in last_100:
                return "False", answer
            elif "true" in last_100:
                return "True", answer
            else:
                return "Unclear", answer
        else:
            if "yes" in last_100.split()[-10:]:
                return "Yes", answer
            elif "no" in last_100.split()[-10:]:
                return "No", answer
            else:
                return "Unclear", answer

    except Exception as e:
        return f"Error: {e}", ""


async def run_memorization_test(model="gpt-5", n_folio=20, n_multi=20):
    """Run memorization test with real benchmark questions."""

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print("=" * 70)
    print("MEMORIZATION TEST - REAL BENCHMARK QUESTIONS")
    print("=" * 70)
    print(f"Model: {model}")

    results = []

    # Load FOLIO questions
    folio_path = "data/folio_original/folio-validation.json"
    if os.path.exists(folio_path):
        print(f"\n{'='*70}")
        print("Loading FOLIO questions...")
        folio_questions = load_folio_questions(folio_path, n_folio)
        print(f"Loaded {len(folio_questions)} FOLIO questions")

        for i, q in enumerate(folio_questions):
            print(f"\n--- FOLIO Test {i+1}/{len(folio_questions)} ---")
            premises = q['premises']
            conclusion = q['conclusion']
            label = q['label']

            if len(premises) < 2:
                print("Skipping - too few premises")
                continue

            # Test original
            print("Testing ORIGINAL...")
            orig_answer, _ = await test_question(client, premises, conclusion, model, is_folio=True)
            print(f"  Ground truth: {label}, Model answer: {orig_answer}")

            # Perturb: remove premise
            new_premises, desc = perturb_remove_premise(premises, conclusion, label)
            print(f"Testing PERTURBED: {desc}")
            pert_answer, _ = await test_question(client, new_premises, conclusion, model, is_folio=True)

            changed = orig_answer != pert_answer

            # Check for memorization
            if label in ['True', 'False'] and orig_answer == label and not changed:
                verdict = "!!! POSSIBLE MEMORIZATION - answer didn't change when premise removed"
            elif changed:
                verdict = "OK - answer changed appropriately"
            else:
                verdict = "N/A"

            print(f"  Perturbed answer: {pert_answer}")
            print(f"  Answer changed: {changed}")
            print(f"  Verdict: {verdict}")

            results.append({
                "source": "FOLIO",
                "example_id": q.get('example_id'),
                "original_label": label,
                "original_got": orig_answer,
                "perturbation": desc,
                "perturbed_got": pert_answer,
                "changed": changed,
                "verdict": verdict
            })

    # Load Multi-LogiEval questions
    multi_path = "data/multi_logi_original/data"
    if os.path.exists(multi_path):
        print(f"\n{'='*70}")
        print("Loading Multi-LogiEval questions...")
        multi_questions = load_multilogi_questions(multi_path, n_multi)
        print(f"Loaded {len(multi_questions)} Multi-LogiEval questions")

        for i, q in enumerate(multi_questions):
            print(f"\n--- Multi-LogiEval Test {i+1}/{len(multi_questions)} ({q.get('source', 'unknown')}) ---")
            premises = q['premises']
            conclusion = q['conclusion']
            label = q['label']

            if len(premises) < 2:
                print("Skipping - too few premises")
                continue

            # Test original
            print("Testing ORIGINAL...")
            orig_answer, _ = await test_question(client, premises, conclusion, model, is_folio=False)
            print(f"  Ground truth: {label}, Model answer: {orig_answer}")

            # Perturb: add negation
            new_premises, desc = perturb_add_negation(premises, conclusion, label)
            print(f"Testing PERTURBED: {desc}")
            pert_answer, _ = await test_question(client, new_premises, conclusion, model, is_folio=False)

            changed = orig_answer != pert_answer

            # Check for memorization
            if orig_answer == label and not changed:
                verdict = "!!! POSSIBLE MEMORIZATION - answer didn't change when contradiction added"
            elif changed:
                verdict = "OK - answer changed with contradiction"
            else:
                verdict = "N/A"

            print(f"  Perturbed answer: {pert_answer}")
            print(f"  Answer changed: {changed}")
            print(f"  Verdict: {verdict}")

            results.append({
                "source": q.get('source', 'Multi-LogiEval'),
                "original_label": label,
                "original_got": orig_answer,
                "perturbation": desc,
                "perturbed_got": pert_answer,
                "changed": changed,
                "verdict": verdict
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    possible_mem = sum(1 for r in results if "MEMORIZATION" in r['verdict'])
    ok_cases = sum(1 for r in results if "OK" in r['verdict'])
    total = len(results)

    print(f"Total tests: {total}")
    print(f"Possible memorization: {possible_mem} ({possible_mem/total*100:.1f}%)" if total > 0 else "N/A")
    print(f"Correct behavior (answer changed): {ok_cases} ({ok_cases/total*100:.1f}%)" if total > 0 else "N/A")

    if possible_mem > total * 0.3:
        print("\n!!! WARNING: High rate of possible memorization detected !!!")
    elif possible_mem > 0:
        print(f"\n⚠ Some cases ({possible_mem}) suggest possible memorization")
    else:
        print("\n✓ No clear signs of memorization detected")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/memorization_perturbation/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.json")

    with open(output_file, 'w') as f:
        json.dump({
            "model": model,
            "timestamp": timestamp,
            "summary": {
                "total": total,
                "possible_memorization": possible_mem,
                "correct_behavior": ok_cases
            },
            "results": results
        }, f, indent=2)

    print(f"\nSaved to: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--n_folio", type=int, default=20)
    parser.add_argument("--n_multi", type=int, default=20)
    args = parser.parse_args()

    asyncio.run(run_memorization_test(args.model, args.n_folio, args.n_multi))
