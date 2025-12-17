#!/usr/bin/env python3
"""
Memorization Test with "Unknown" Option for Multi-LogiEval

The original Multi-LogiEval only has Yes/No answers. By adding "Unknown" as a third option,
we can better detect memorization:

1. When premises are removed: A reasoning model should say "Unknown" (insufficient info)
2. When contradiction is added: A reasoning model should say "Unknown" (contradictory info)

If the model still answers Yes/No matching the original answer, it suggests memorization.

This test focuses on PERTURBED questions only (not original) since we want to see
if the model recognizes when it CANNOT answer definitively.
"""

import argparse
import asyncio
import random
import re
import json
import os
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

from utils.answer_parsing import normalize_answer
from datasets.folio import load_folio
from datasets.multilogieval import load_and_sample_multilogieval


def perturb_remove_premise(premises_list, conclusion):
    """Remove the premise most related to conclusion (by word overlap)."""
    if len(premises_list) <= 1:
        return premises_list, None

    conclusion_words = set(conclusion.lower().split())

    best_idx = 0
    best_overlap = 0
    for i, p in enumerate(premises_list):
        p_words = set(p.lower().split())
        overlap = len(p_words & conclusion_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i

    removed = premises_list[best_idx]
    new_premises = premises_list[:best_idx] + premises_list[best_idx+1:]
    return new_premises, removed


def perturb_add_negation(conclusion):
    """Create negation of conclusion to add as contradictory premise."""
    c = conclusion.lower()

    if "is a" in c:
        return conclusion.replace(" is a ", " is not a ").replace(" Is a ", " Is not a ")
    elif "is " in c:
        return conclusion.replace(" is ", " is not ", 1).replace(" Is ", " Is not ", 1)
    elif "can " in c:
        return conclusion.replace(" can ", " cannot ", 1).replace(" Can ", " Cannot ", 1)
    elif "will " in c:
        return conclusion.replace(" will ", " will not ", 1).replace(" Will ", " Will not ", 1)
    elif "does " in c:
        return conclusion.replace(" does ", " does not ", 1).replace(" Does ", " Does not ", 1)
    else:
        return "It is NOT the case that " + conclusion


def parse_yes_no_unknown(response_text):
    """Parse Yes/No/Unknown from response."""
    text = response_text.lower().strip()
    last_100 = text[-100:]

    # Check for Unknown first (since it's the new option we're testing)
    if "unknown" in last_100 or "cannot determine" in last_100 or "insufficient" in last_100:
        return "Unknown"
    elif "yes" in last_100.split()[-10:]:
        return "Yes"
    elif "no" in last_100.split()[-10:]:
        return "No"
    else:
        # Try to find any answer
        if "unknown" in text:
            return "Unknown"
        elif "yes" in text:
            return "Yes"
        elif "no" in text:
            return "No"
        return "Unclear"


async def test_multilogi_with_unknown(sample, client, model, perturbation_type, semaphore):
    """
    Test Multi-LogiEval question with Yes/No/Unknown options.

    Tests BOTH original and perturbed versions.
    For perturbed: A good model should answer "Unknown" since the information is insufficient/contradictory.
    """
    async with semaphore:
        context = sample.get('context', '')
        question = sample.get('question', '')
        answer = sample.get('answer', '')

        # Split context into sentences
        sentences = [s.strip() + '.' for s in re.split(r'\.(?:\s|$)', context) if s.strip()]

        if len(sentences) < 2:
            return {'skip': True, 'reason': 'too few sentences'}

        ground_truth = normalize_answer(answer, answer_format="yes_no")

        # System prompt emphasizing Unknown option
        system_prompt = """You are a logical reasoning expert. Answer based ONLY on the given context.

IMPORTANT: You must answer with exactly one of:
- Yes: The statement can be proven TRUE from the context
- No: The statement can be proven FALSE from the context
- Unknown: The context is INSUFFICIENT or CONTRADICTORY to determine the answer

If any critical information is missing, or if the premises contradict each other, answer Unknown."""

        # Test ORIGINAL
        orig_prompt = f"""Context: {context}

Question: {question}

Based ONLY on the context above, answer: Yes, No, or Unknown"""

        try:
            orig_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": orig_prompt}
                ]
            )
            orig_answer = parse_yes_no_unknown(orig_resp.choices[0].message.content)

            # Test PERTURBED
            if perturbation_type == 'remove_premise':
                new_sentences, removed = perturb_remove_premise(sentences, question)
                perturbed_context = ' '.join(new_sentences)
                detail = removed[:80] if removed else None
            else:  # add_contradiction
                negation = perturb_add_negation(question)
                perturbed_context = context + " " + negation
                detail = negation[:80]

            pert_prompt = f"""Context: {perturbed_context}

Question: {question}

Based ONLY on the context above, answer: Yes, No, or Unknown"""

            pert_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": pert_prompt}
                ]
            )
            pert_answer = parse_yes_no_unknown(pert_resp.choices[0].message.content)

            # Analysis
            orig_correct = orig_answer == ground_truth
            pert_says_unknown = pert_answer == "Unknown"
            pert_matches_original = pert_answer == orig_answer
            pert_matches_ground_truth = pert_answer == ground_truth

            # Verdict logic:
            # - If perturbed says Unknown: GOOD (model recognizes uncertainty)
            # - If perturbed still matches ground truth: POSSIBLE MEMORIZATION
            # - If perturbed changed (but not to Unknown): PARTIAL (model noticed something wrong)
            if pert_says_unknown:
                verdict = "GOOD_UNKNOWN"
            elif pert_matches_ground_truth and orig_correct:
                verdict = "POSSIBLE_MEMORIZATION"
            elif not pert_matches_original:
                verdict = "CHANGED_BUT_NOT_UNKNOWN"
            else:
                verdict = "N/A"

            return {
                'logic_type': sample.get('logic_type'),
                'depth': sample.get('depth'),
                'rule': sample.get('rule'),
                'perturbation': perturbation_type,
                'ground_truth': ground_truth,
                'orig_answer': orig_answer,
                'orig_correct': orig_correct,
                'pert_answer': pert_answer,
                'pert_says_unknown': pert_says_unknown,
                'pert_matches_gt': pert_matches_ground_truth,
                'verdict': verdict,
                'detail': detail
            }

        except Exception as e:
            return {'error': str(e)}


async def test_folio_with_unknown(example, client, model, perturbation_type, semaphore):
    """Test FOLIO question - already has Unknown option."""
    async with semaphore:
        premises_str = example.get('premises', '')
        premises_list = [p.strip() for p in premises_str.split('\n') if p.strip()]
        conclusion = example.get('conclusion', '')
        label = example.get('label', '')

        if len(premises_list) < 2:
            return {'skip': True, 'reason': 'too few premises'}

        ground_truth = normalize_answer(label, answer_format="true_false")

        system_prompt = """You are a logical reasoning assistant. Determine if conclusions follow from premises.

IMPORTANT: You must answer with exactly one of:
- True: The conclusion can be PROVEN from the premises
- False: The conclusion can be DISPROVEN from the premises
- Unknown: The premises are INSUFFICIENT or CONTRADICTORY to determine the answer

If any critical information is missing, or if the premises contradict each other, answer Unknown."""

        # Test ORIGINAL
        orig_prompt = f"""Premises:
{chr(10).join('- ' + p for p in premises_list)}

Conclusion: {conclusion}

Based on the premises, is the conclusion True, False, or Unknown?"""

        try:
            orig_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": orig_prompt}
                ]
            )
            orig_text = orig_resp.choices[0].message.content.lower()
            if "unknown" in orig_text[-100:]:
                orig_answer = "Unknown"
            elif "false" in orig_text[-100:]:
                orig_answer = "False"
            elif "true" in orig_text[-100:]:
                orig_answer = "True"
            else:
                orig_answer = "Unclear"

            # Test PERTURBED
            if perturbation_type == 'remove_premise':
                new_premises, removed = perturb_remove_premise(premises_list, conclusion)
                detail = removed[:80] if removed else None
            else:  # add_contradiction
                negation = perturb_add_negation(conclusion)
                new_premises = premises_list + [negation]
                detail = negation[:80]

            pert_prompt = f"""Premises:
{chr(10).join('- ' + p for p in new_premises)}

Conclusion: {conclusion}

Based on the premises, is the conclusion True, False, or Unknown?"""

            pert_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": pert_prompt}
                ]
            )
            pert_text = pert_resp.choices[0].message.content.lower()
            if "unknown" in pert_text[-100:]:
                pert_answer = "Unknown"
            elif "false" in pert_text[-100:]:
                pert_answer = "False"
            elif "true" in pert_text[-100:]:
                pert_answer = "True"
            else:
                pert_answer = "Unclear"

            # Analysis
            orig_correct = orig_answer == ground_truth
            pert_says_unknown = pert_answer == "Unknown"
            pert_matches_gt = pert_answer == ground_truth

            # Verdict logic for FOLIO:
            # - For True/False ground truth:
            #   - If perturbed says Unknown: GOOD (recognizes uncertainty)
            #   - If perturbed matches GT and was correct: POSSIBLE_MEMORIZATION
            # - For Unknown ground truth:
            #   - If perturbed says Unknown: Could be either reasoning or memorization
            #   - Special category: GT_UNKNOWN_STAYED_UNKNOWN
            if ground_truth == "Unknown":
                if pert_says_unknown and orig_correct:
                    verdict = "GT_UNKNOWN_STAYED_UNKNOWN"  # Ambiguous - could be memorization
                elif pert_answer != orig_answer:
                    verdict = "GT_UNKNOWN_CHANGED"  # Interesting - perturbation changed answer
                else:
                    verdict = "GT_UNKNOWN_OTHER"
            else:  # True/False ground truth
                if pert_says_unknown:
                    verdict = "GOOD_UNKNOWN"
                elif pert_matches_gt and orig_correct:
                    verdict = "POSSIBLE_MEMORIZATION"
                elif pert_answer != orig_answer:
                    verdict = "CHANGED_BUT_NOT_UNKNOWN"
                else:
                    verdict = "N/A"

            return {
                'example_id': example.get('example_id'),
                'story_id': example.get('story_id'),
                'perturbation': perturbation_type,
                'ground_truth': ground_truth,
                'orig_answer': orig_answer,
                'orig_correct': orig_correct,
                'pert_answer': pert_answer,
                'pert_says_unknown': pert_says_unknown,
                'pert_matches_gt': pert_matches_gt,
                'verdict': verdict,
                'detail': detail
            }

        except Exception as e:
            return {'example_id': example.get('example_id'), 'error': str(e)}


def print_results(name, results):
    """Print results with Unknown analysis."""
    valid = [r for r in results if 'error' not in r and not r.get('skip')]
    if not valid:
        print(f"  No valid results")
        return valid

    total = len(valid)
    good_unknown = sum(1 for r in valid if r['verdict'] == 'GOOD_UNKNOWN')
    possible_mem = sum(1 for r in valid if r['verdict'] == 'POSSIBLE_MEMORIZATION')
    changed_not_unknown = sum(1 for r in valid if r['verdict'] == 'CHANGED_BUT_NOT_UNKNOWN')
    orig_correct = sum(1 for r in valid if r.get('orig_correct', False))

    # Breakdown by ground truth (for FOLIO with Unknown)
    gt_unknown_stayed = sum(1 for r in valid if r['verdict'] == 'GT_UNKNOWN_STAYED_UNKNOWN')
    gt_unknown_changed = sum(1 for r in valid if r['verdict'] == 'GT_UNKNOWN_CHANGED')
    gt_unknown_other = sum(1 for r in valid if r['verdict'] == 'GT_UNKNOWN_OTHER')

    # Count by ground truth
    by_gt = {}
    for r in valid:
        gt = r.get('ground_truth', 'N/A')
        by_gt[gt] = by_gt.get(gt, 0) + 1

    print(f"\n{name} Results:")
    print(f"  Total valid: {total}")
    print(f"  Original accuracy: {orig_correct}/{total} ({orig_correct/total*100:.1f}%)")

    # Show breakdown by ground truth
    print(f"  By ground truth: {by_gt}")

    # For True/False GT cases
    tf_total = sum(1 for r in valid if r.get('ground_truth') in ['True', 'False', 'Yes', 'No'])
    if tf_total > 0:
        print(f"\n  True/False GT cases ({tf_total}):")
        print(f"    Perturbed says Unknown: {good_unknown}/{tf_total} ({good_unknown/tf_total*100:.1f}%) - GOOD")
        print(f"    Possible memorization: {possible_mem}/{tf_total} ({possible_mem/tf_total*100:.1f}%) - SUSPICIOUS")
        print(f"    Changed but not Unknown: {changed_not_unknown}/{tf_total} ({changed_not_unknown/tf_total*100:.1f}%)")

    # For Unknown GT cases (FOLIO only)
    unknown_total = gt_unknown_stayed + gt_unknown_changed + gt_unknown_other
    if unknown_total > 0:
        print(f"\n  Unknown GT cases ({unknown_total}):")
        print(f"    Stayed Unknown: {gt_unknown_stayed}/{unknown_total} ({gt_unknown_stayed/unknown_total*100:.1f}%) - AMBIGUOUS")
        print(f"    Changed from Unknown: {gt_unknown_changed}/{unknown_total} ({gt_unknown_changed/unknown_total*100:.1f}%) - INTERESTING")
        print(f"    Other: {gt_unknown_other}/{unknown_total}")

    return valid


async def main_async():
    parser = argparse.ArgumentParser(description='Memorization test with Unknown option')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--model', default='gpt-4o', help='Model to test')
    parser.add_argument('--dataset', choices=['folio', 'multilogieval', 'both'], default='both')
    parser.add_argument('--folio_file', default='data/folio_original/folio-validation.json')
    parser.add_argument('--multi_dir', default='data/multi_logi_original/data')
    parser.add_argument('--num_samples', type=int, default=50, help='Samples per dataset')
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    client = AsyncOpenAI(api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    print("=" * 70)
    print("MEMORIZATION TEST WITH UNKNOWN OPTION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Key insight: When premises are removed or contradicted,")
    print(f"             a reasoning model should say 'Unknown'.")
    print(f"             If it still gives the original answer, that's suspicious.")
    print("=" * 70)

    all_results = {
        'folio_remove_premise': [],
        'folio_add_contradiction': [],
        'multilogi_remove_premise': [],
        'multilogi_add_contradiction': []
    }

    # Test FOLIO
    if args.dataset in ['folio', 'both']:
        folio_data = load_folio(args.folio_file)
        # Include ALL labels including Unknown/Uncertain (34% of FOLIO)
        # Note: FOLIO uses "Uncertain" but we normalize to "Unknown"

        if args.num_samples > 0 and args.num_samples < len(folio_data):
            folio_data = random.sample(folio_data, args.num_samples)

        print(f"\n{'='*70}")
        print(f"FOLIO: Testing {len(folio_data)} questions")
        print(f"{'='*70}")

        # Test Remove Premise
        print(f"\n--- FOLIO: Remove Critical Premise ---")
        tasks = [test_folio_with_unknown(ex, client, args.model, 'remove_premise', semaphore) for ex in folio_data]
        results = await tqdm_asyncio.gather(*tasks, desc="FOLIO-Remove")
        all_results['folio_remove_premise'] = print_results("FOLIO Remove Premise", results)

        # Test Add Contradiction
        print(f"\n--- FOLIO: Add Contradiction ---")
        tasks = [test_folio_with_unknown(ex, client, args.model, 'add_contradiction', semaphore) for ex in folio_data]
        results = await tqdm_asyncio.gather(*tasks, desc="FOLIO-Contradict")
        all_results['folio_add_contradiction'] = print_results("FOLIO Add Contradiction", results)

    # Test Multi-LogiEval
    if args.dataset in ['multilogieval', 'both']:
        multi_data = load_and_sample_multilogieval(
            args.multi_dir,
            samples_per_combination=max(3, args.num_samples // 15),
            seed=args.seed
        )

        if args.num_samples > 0 and args.num_samples < len(multi_data):
            multi_data = random.sample(multi_data, args.num_samples)

        print(f"\n{'='*70}")
        print(f"Multi-LogiEval: Testing {len(multi_data)} questions WITH UNKNOWN OPTION")
        print(f"{'='*70}")

        # Test Remove Premise
        print(f"\n--- Multi-LogiEval: Remove Critical Premise ---")
        tasks = [test_multilogi_with_unknown(s, client, args.model, 'remove_premise', semaphore) for s in multi_data]
        results = await tqdm_asyncio.gather(*tasks, desc="Multi-Remove")
        all_results['multilogi_remove_premise'] = print_results("Multi-LogiEval Remove Premise", results)

        # Test Add Contradiction
        print(f"\n--- Multi-LogiEval: Add Contradiction ---")
        tasks = [test_multilogi_with_unknown(s, client, args.model, 'add_contradiction', semaphore) for s in multi_data]
        results = await tqdm_asyncio.gather(*tasks, desc="Multi-Contradict")
        all_results['multilogi_add_contradiction'] = print_results("Multi-LogiEval Add Contradiction", results)

    # Summary
    print("\n" + "=" * 70)
    print("MEMORIZATION TEST SUMMARY (WITH UNKNOWN OPTION)")
    print("=" * 70)

    summary_table = []
    for key, results in all_results.items():
        if results:
            total = len(results)
            good = sum(1 for r in results if r['verdict'] == 'GOOD_UNKNOWN')
            mem = sum(1 for r in results if r['verdict'] == 'POSSIBLE_MEMORIZATION')
            changed = sum(1 for r in results if r['verdict'] == 'CHANGED_BUT_NOT_UNKNOWN')
            summary_table.append({
                'test': key,
                'total': total,
                'good_unknown': good,
                'good_pct': good/total*100 if total > 0 else 0,
                'possible_mem': mem,
                'mem_pct': mem/total*100 if total > 0 else 0,
                'changed': changed
            })

    print(f"\n{'Test':<30} {'Total':>6} {'Unknown':>8} {'%':>6} {'Mem?':>6} {'%':>6} {'Changed':>8}")
    print("-" * 80)
    for row in summary_table:
        print(f"{row['test']:<30} {row['total']:>6} {row['good_unknown']:>8} {row['good_pct']:>5.1f}% {row['possible_mem']:>6} {row['mem_pct']:>5.1f}% {row['changed']:>8}")

    total_all = sum(r['total'] for r in summary_table)
    total_unknown = sum(r['good_unknown'] for r in summary_table)
    total_mem = sum(r['possible_mem'] for r in summary_table)

    print("-" * 80)
    print(f"{'TOTAL':<30} {total_all:>6} {total_unknown:>8} {total_unknown/total_all*100 if total_all > 0 else 0:>5.1f}% {total_mem:>6} {total_mem/total_all*100 if total_all > 0 else 0:>5.1f}%")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"  'Unknown' responses: {total_unknown}/{total_all} ({total_unknown/total_all*100:.1f}%)")
    print(f"    → Model recognizes when info is insufficient/contradictory")
    print(f"  'Possible memorization': {total_mem}/{total_all} ({total_mem/total_all*100:.1f}%)")
    print(f"    → Model gives original answer despite perturbation")

    if total_unknown / total_all > 0.5:
        print("\n✓ Good! Model frequently recognizes uncertainty after perturbation.")
    elif total_mem / total_all > 0.3:
        print("\n!!! WARNING: High rate of possible memorization detected !!!")
    else:
        print("\n⚠ Mixed results. Model sometimes recognizes uncertainty.")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/memorization_unknown/{args.model}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump({
            'model': args.model,
            'timestamp': timestamp,
            'num_samples': args.num_samples,
            'summary': {
                'total': total_all,
                'good_unknown': total_unknown,
                'possible_memorization': total_mem,
                'by_test': summary_table
            },
            'folio_remove_premise': all_results['folio_remove_premise'],
            'folio_add_contradiction': all_results['folio_add_contradiction'],
            'multilogi_remove_premise': all_results['multilogi_remove_premise'],
            'multilogi_add_contradiction': all_results['multilogi_add_contradiction']
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}/results.json")


if __name__ == "__main__":
    asyncio.run(main_async())
