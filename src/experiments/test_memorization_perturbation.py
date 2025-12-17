#!/usr/bin/env python3
"""
Memorization Test via Perturbation

Tests if the model memorized benchmarks by applying TWO perturbation types to EACH dataset:
1. Remove critical premise → answer should become Unknown
2. Add contradictory premise → answer should flip

For each dataset (FOLIO and Multi-LogiEval), we test both perturbations.

Key insight: If model memorized the dataset, it will give the ORIGINAL answer
even when the question is modified in ways that should change the answer.
"""

import argparse
import asyncio
import random
import re
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

from utils.answer_parsing import parse_folio_answer, parse_multilogieval_answer, normalize_answer
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


async def test_folio_remove_premise(example, client, model, semaphore):
    """Test FOLIO question: original vs perturbed (premise removed)."""
    async with semaphore:
        premises_str = example.get('premises', '')
        premises_list = [p.strip() for p in premises_str.split('\n') if p.strip()]
        conclusion = example.get('conclusion', '')
        label = example.get('label', '')

        if len(premises_list) < 2:
            return {'skip': True, 'reason': 'too few premises'}

        system_prompt = "You are a logical reasoning assistant. Determine if conclusions follow from premises."

        orig_prompt = f"""Premises:
{chr(10).join('- ' + p for p in premises_list)}

Conclusion: {conclusion}

Based on the premises, is the conclusion True, False, or Unknown?
Answer with exactly one of: True, False, or Unknown"""

        try:
            orig_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": orig_prompt}
                ]
            )
            orig_answer = parse_folio_answer(orig_resp.choices[0].message.content)

            # Test PERTURBED (remove critical premise)
            new_premises, removed = perturb_remove_premise(premises_list, conclusion)

            pert_prompt = f"""Premises:
{chr(10).join('- ' + p for p in new_premises)}

Conclusion: {conclusion}

Based on the premises, is the conclusion True, False, or Unknown?
Answer with exactly one of: True, False, or Unknown"""

            pert_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": pert_prompt}
                ]
            )
            pert_answer = parse_folio_answer(pert_resp.choices[0].message.content)

            changed = orig_answer != pert_answer
            ground_truth = normalize_answer(label, answer_format="true_false")

            if ground_truth in ['True', 'False'] and orig_answer == ground_truth and not changed:
                verdict = "POSSIBLE_MEMORIZATION"
            elif changed:
                verdict = "OK"
            else:
                verdict = "N/A"

            return {
                'example_id': example.get('example_id'),
                'story_id': example.get('story_id'),
                'perturbation': 'remove_premise',
                'ground_truth': ground_truth,
                'orig_answer': orig_answer,
                'pert_answer': pert_answer,
                'changed': changed,
                'verdict': verdict,
                'detail': removed[:80] if removed else None
            }

        except Exception as e:
            return {'example_id': example.get('example_id'), 'error': str(e)}


async def test_folio_add_contradiction(example, client, model, semaphore):
    """Test FOLIO question: original vs perturbed (contradiction added)."""
    async with semaphore:
        premises_str = example.get('premises', '')
        premises_list = [p.strip() for p in premises_str.split('\n') if p.strip()]
        conclusion = example.get('conclusion', '')
        label = example.get('label', '')

        if len(premises_list) < 1:
            return {'skip': True, 'reason': 'no premises'}

        system_prompt = "You are a logical reasoning assistant. Determine if conclusions follow from premises."

        orig_prompt = f"""Premises:
{chr(10).join('- ' + p for p in premises_list)}

Conclusion: {conclusion}

Based on the premises, is the conclusion True, False, or Unknown?
Answer with exactly one of: True, False, or Unknown"""

        try:
            orig_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": orig_prompt}
                ]
            )
            orig_answer = parse_folio_answer(orig_resp.choices[0].message.content)

            # Test PERTURBED (add contradiction of conclusion)
            negation = perturb_add_negation(conclusion)
            new_premises = premises_list + [negation]

            pert_prompt = f"""Premises:
{chr(10).join('- ' + p for p in new_premises)}

Conclusion: {conclusion}

Based on the premises, is the conclusion True, False, or Unknown?
Answer with exactly one of: True, False, or Unknown"""

            pert_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": pert_prompt}
                ]
            )
            pert_answer = parse_folio_answer(pert_resp.choices[0].message.content)

            changed = orig_answer != pert_answer
            ground_truth = normalize_answer(label, answer_format="true_false")

            # For contradiction: if original was True and model still says True, possible memorization
            if ground_truth in ['True', 'False'] and orig_answer == ground_truth and not changed:
                verdict = "POSSIBLE_MEMORIZATION"
            elif changed:
                verdict = "OK"
            else:
                verdict = "N/A"

            return {
                'example_id': example.get('example_id'),
                'story_id': example.get('story_id'),
                'perturbation': 'add_contradiction',
                'ground_truth': ground_truth,
                'orig_answer': orig_answer,
                'pert_answer': pert_answer,
                'changed': changed,
                'verdict': verdict,
                'detail': negation[:80]
            }

        except Exception as e:
            return {'example_id': example.get('example_id'), 'error': str(e)}


async def test_multilogi_add_contradiction(sample, client, model, semaphore):
    """Test Multi-LogiEval question: original vs perturbed (contradiction added)."""
    async with semaphore:
        context = sample.get('context', '')
        question = sample.get('question', '')
        answer = sample.get('answer', '')

        sentences = [s.strip() + '.' for s in re.split(r'\.(?:\s|$)', context) if s.strip()]

        if len(sentences) < 2:
            return {'skip': True, 'reason': 'too few sentences'}

        system_prompt = "You are a logical reasoning expert. Answer based only on the given context."

        orig_prompt = f"""Context: {context}

Question: {question}

Answer with exactly: Yes or No"""

        try:
            orig_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": orig_prompt}
                ]
            )
            orig_answer = parse_multilogieval_answer(orig_resp.choices[0].message.content)

            # Test PERTURBED (add contradiction)
            negation = perturb_add_negation(question)
            perturbed_context = context + " " + negation

            pert_prompt = f"""Context: {perturbed_context}

Question: {question}

Answer with exactly: Yes or No"""

            pert_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": pert_prompt}
                ]
            )
            pert_answer = parse_multilogieval_answer(pert_resp.choices[0].message.content)

            changed = orig_answer != pert_answer
            ground_truth = normalize_answer(answer, answer_format="yes_no")

            if orig_answer == ground_truth and not changed:
                verdict = "POSSIBLE_MEMORIZATION"
            elif changed:
                verdict = "OK"
            else:
                verdict = "N/A"

            return {
                'logic_type': sample.get('logic_type'),
                'depth': sample.get('depth'),
                'rule': sample.get('rule'),
                'perturbation': 'add_contradiction',
                'ground_truth': ground_truth,
                'orig_answer': orig_answer,
                'pert_answer': pert_answer,
                'changed': changed,
                'verdict': verdict,
                'detail': negation[:80]
            }

        except Exception as e:
            return {'error': str(e)}


async def test_multilogi_remove_premise(sample, client, model, semaphore):
    """Test Multi-LogiEval question: original vs perturbed (premise removed)."""
    async with semaphore:
        context = sample.get('context', '')
        question = sample.get('question', '')
        answer = sample.get('answer', '')

        # Split context into sentences (premises)
        sentences = [s.strip() + '.' for s in re.split(r'\.(?:\s|$)', context) if s.strip()]

        if len(sentences) < 2:
            return {'skip': True, 'reason': 'too few sentences'}

        system_prompt = "You are a logical reasoning expert. Answer based only on the given context."

        orig_prompt = f"""Context: {context}

Question: {question}

Answer with exactly: Yes or No"""

        try:
            orig_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": orig_prompt}
                ]
            )
            orig_answer = parse_multilogieval_answer(orig_resp.choices[0].message.content)

            # Test PERTURBED (remove critical premise)
            new_sentences, removed = perturb_remove_premise(sentences, question)
            perturbed_context = ' '.join(new_sentences)

            pert_prompt = f"""Context: {perturbed_context}

Question: {question}

Answer with exactly: Yes or No"""

            pert_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": pert_prompt}
                ]
            )
            pert_answer = parse_multilogieval_answer(pert_resp.choices[0].message.content)

            changed = orig_answer != pert_answer
            ground_truth = normalize_answer(answer, answer_format="yes_no")

            if orig_answer == ground_truth and not changed:
                verdict = "POSSIBLE_MEMORIZATION"
            elif changed:
                verdict = "OK"
            else:
                verdict = "N/A"

            return {
                'logic_type': sample.get('logic_type'),
                'depth': sample.get('depth'),
                'rule': sample.get('rule'),
                'perturbation': 'remove_premise',
                'ground_truth': ground_truth,
                'orig_answer': orig_answer,
                'pert_answer': pert_answer,
                'changed': changed,
                'verdict': verdict,
                'detail': removed[:80] if removed else None
            }

        except Exception as e:
            return {'error': str(e)}


def print_results(name, results):
    """Print results for a perturbation test."""
    valid = [r for r in results if 'error' not in r and not r.get('skip')]
    if not valid:
        print(f"  No valid results")
        return valid

    possible_mem = sum(1 for r in valid if r['verdict'] == 'POSSIBLE_MEMORIZATION')
    ok_cases = sum(1 for r in valid if r['verdict'] == 'OK')
    changed = sum(1 for r in valid if r['changed'])

    print(f"\n{name} Results:")
    print(f"  Total valid: {len(valid)}")
    print(f"  Answer changed: {changed}/{len(valid)} ({changed/len(valid)*100:.1f}%)")
    print(f"  Possible memorization: {possible_mem} ({possible_mem/len(valid)*100:.1f}%)")
    print(f"  OK (answer changed): {ok_cases}")

    return valid


async def main_async():
    parser = argparse.ArgumentParser(description='Memorization test via perturbation')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--model', default='gpt-4o', help='Model to test')
    parser.add_argument('--dataset', choices=['folio', 'multilogieval', 'both'], default='both')
    parser.add_argument('--folio_file', default='data/folio_original/folio-validation.json')
    parser.add_argument('--multi_dir', default='data/multi_logi_original/data')
    parser.add_argument('--num_samples', type=int, default=50, help='Samples per dataset per perturbation type')
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    client = AsyncOpenAI(api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    print("=" * 70)
    print("MEMORIZATION TEST VIA PERTURBATION (EXPANDED)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Testing BOTH perturbation types on BOTH datasets:")
    print(f"  1. Remove critical premise")
    print(f"  2. Add contradiction")
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
        folio_data = [ex for ex in folio_data if ex.get('label') in ['True', 'False']]

        if args.num_samples > 0 and args.num_samples < len(folio_data):
            folio_data = random.sample(folio_data, args.num_samples)

        print(f"\n{'='*70}")
        print(f"FOLIO: Testing {len(folio_data)} questions with BOTH perturbations")
        print(f"{'='*70}")

        # Test 1: Remove premise
        print(f"\n--- FOLIO: Remove Critical Premise ---")
        tasks = [test_folio_remove_premise(ex, client, args.model, semaphore) for ex in folio_data]
        results = await tqdm_asyncio.gather(*tasks, desc="FOLIO-RemovePremise")
        all_results['folio_remove_premise'] = print_results("FOLIO Remove Premise", results)

        # Test 2: Add contradiction
        print(f"\n--- FOLIO: Add Contradiction ---")
        tasks = [test_folio_add_contradiction(ex, client, args.model, semaphore) for ex in folio_data]
        results = await tqdm_asyncio.gather(*tasks, desc="FOLIO-AddContradiction")
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
        print(f"Multi-LogiEval: Testing {len(multi_data)} questions with BOTH perturbations")
        print(f"{'='*70}")

        # Test 1: Remove premise
        print(f"\n--- Multi-LogiEval: Remove Critical Premise ---")
        tasks = [test_multilogi_remove_premise(s, client, args.model, semaphore) for s in multi_data]
        results = await tqdm_asyncio.gather(*tasks, desc="Multi-RemovePremise")
        all_results['multilogi_remove_premise'] = print_results("Multi-LogiEval Remove Premise", results)

        # Test 2: Add contradiction
        print(f"\n--- Multi-LogiEval: Add Contradiction ---")
        tasks = [test_multilogi_add_contradiction(s, client, args.model, semaphore) for s in multi_data]
        results = await tqdm_asyncio.gather(*tasks, desc="Multi-AddContradiction")
        all_results['multilogi_add_contradiction'] = print_results("Multi-LogiEval Add Contradiction", results)

    # Summary
    print("\n" + "=" * 70)
    print("MEMORIZATION TEST SUMMARY")
    print("=" * 70)

    summary_table = []
    for key, results in all_results.items():
        if results:
            total = len(results)
            mem = sum(1 for r in results if r['verdict'] == 'POSSIBLE_MEMORIZATION')
            ok = sum(1 for r in results if r['verdict'] == 'OK')
            summary_table.append({
                'test': key,
                'total': total,
                'possible_mem': mem,
                'mem_rate': mem/total*100 if total > 0 else 0,
                'ok': ok
            })

    print(f"\n{'Test':<35} {'Total':>6} {'Poss.Mem':>10} {'Rate':>8} {'OK':>6}")
    print("-" * 70)
    for row in summary_table:
        print(f"{row['test']:<35} {row['total']:>6} {row['possible_mem']:>10} {row['mem_rate']:>7.1f}% {row['ok']:>6}")

    total_all = sum(r['total'] for r in summary_table)
    total_mem = sum(r['possible_mem'] for r in summary_table)
    total_ok = sum(r['ok'] for r in summary_table)

    print("-" * 70)
    print(f"{'TOTAL':<35} {total_all:>6} {total_mem:>10} {total_mem/total_all*100 if total_all > 0 else 0:>7.1f}% {total_ok:>6}")

    if total_mem > total_all * 0.3:
        print("\n!!! WARNING: High rate of possible memorization detected !!!")
    elif total_mem > 0:
        print(f"\n⚠ Some cases ({total_mem}) suggest possible memorization")
    else:
        print("\n✓ No clear signs of memorization detected")

    # Save results
    import json
    import os
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/memorization_perturbation/{args.model}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump({
            'model': args.model,
            'timestamp': timestamp,
            'num_samples': args.num_samples,
            'summary': {
                'total': total_all,
                'possible_memorization': total_mem,
                'ok_changed': total_ok,
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
