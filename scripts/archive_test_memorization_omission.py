#!/usr/bin/env python3
"""
Memorization test via premise omission.

This is a direct test: if we remove ALL premises and only give the conclusion,
a non-memorized model should answer "Unknown" (or random guess ~33%/50%).
If the model still achieves high accuracy, it likely memorized the dataset.

Test modes:
1. keep_none: Remove ALL premises - only show conclusion
2. keep_one: Keep only 1 random premise
3. conclusion_only: Only show "Based on general knowledge, is X true?"
"""

import json
import asyncio
import random
import os
import argparse
from datetime import datetime
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

random.seed(42)


def load_folio(filepath):
    """Load FOLIO dataset."""
    with open(filepath) as f:
        return json.load(f)


def load_multilogieval(data_dir, max_per_file=5):
    """Load Multi-LogiEval with stratified sampling."""
    samples = []

    for depth_dir in sorted(os.listdir(data_dir)):
        if not depth_dir.startswith('d'):
            continue
        depth_path = os.path.join(data_dir, depth_dir)
        if not os.path.isdir(depth_path):
            continue

        for logic_type in ['fol', 'nm', 'pl']:
            logic_path = os.path.join(depth_path, logic_type)
            if not os.path.isdir(logic_path):
                continue

            for fname in os.listdir(logic_path):
                if not fname.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(logic_path, fname)) as f:
                        data = json.load(f)

                    file_samples = data.get('samples', [])[:max_per_file]
                    for s in file_samples:
                        samples.append({
                            'context': s.get('context', ''),
                            'question': s.get('question', ''),
                            'answer': s.get('answer', ''),
                            'logic': logic_type,
                            'depth': depth_dir,
                            'rule': data.get('rule', fname.replace('.json', ''))
                        })
                except Exception:
                    continue

    return samples


async def test_folio_question(client, question, mode, model, semaphore):
    """Test a single FOLIO question with premise omission."""
    async with semaphore:
        premises = question.get('premises', '')
        conclusion = question.get('conclusion', '')
        label = question.get('label', '')

        # Apply omission strategy
        if mode == 'keep_none':
            test_premises = "[Premises removed for testing]"
        elif mode == 'keep_one':
            sentences = [s.strip() for s in premises.split('\n') if s.strip()]
            test_premises = random.choice(sentences) if sentences else "[No premises]"
        elif mode == 'conclusion_only':
            test_premises = None
        else:
            test_premises = premises

        if mode == 'conclusion_only':
            prompt = f"""Is the following statement True, False, or Unknown?
Answer based on your general knowledge.

Statement: {conclusion}

Answer with exactly one of: True, False, or Unknown"""
        else:
            prompt = f"""Premises:
{test_premises}

Conclusion: {conclusion}

Based on the premises above, is the conclusion True, False, or Unknown?
Answer with exactly one of: True, False, or Unknown"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a logical reasoning assistant. Determine if conclusions follow from premises."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0 if 'gpt-4' in model else None
            )

            answer = response.choices[0].message.content.strip().lower()

            # Parse answer
            if 'unknown' in answer[-50:] or 'uncertain' in answer[-50:]:
                prediction = 'Unknown'
            elif 'false' in answer[-50:]:
                prediction = 'False'
            elif 'true' in answer[-50:]:
                prediction = 'True'
            else:
                prediction = 'Unknown'

            # Normalize ground truth
            gt = label.lower() if label else 'unknown'
            if gt in ['true', 'yes']:
                gt = 'True'
            elif gt in ['false', 'no']:
                gt = 'False'
            else:
                gt = 'Unknown'

            return {
                'example_id': question.get('example_id'),
                'conclusion': conclusion,
                'ground_truth': gt,
                'prediction': prediction,
                'correct': prediction == gt,
                'mode': mode
            }
        except Exception as e:
            return {'example_id': question.get('example_id'), 'error': str(e)}


async def test_multilogi_question(client, sample, mode, model, semaphore):
    """Test a single Multi-LogiEval question with context omission."""
    async with semaphore:
        context = sample.get('context', '')
        question = sample.get('question', '')
        answer = sample.get('answer', '')

        # Apply omission
        if mode == 'keep_none':
            test_context = "[Context removed for testing]"
        elif mode == 'keep_one':
            import re
            sentences = [s.strip() + '.' for s in re.split(r'\.(?:\s|$)', context) if s.strip()]
            test_context = random.choice(sentences) if sentences else "[No context]"
        elif mode == 'conclusion_only':
            test_context = None
        else:
            test_context = context

        if mode == 'conclusion_only':
            prompt = f"""Answer this question based on your general knowledge.
Question: {question}
Answer with exactly: Yes or No"""
        else:
            prompt = f"""Context: {test_context}

Question: {question}

Based on the context, answer with exactly: Yes or No"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a logical reasoning expert. Answer based only on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0 if 'gpt-4' in model else None
            )

            resp = response.choices[0].message.content.strip().lower()

            # Parse
            if 'yes' in resp[-20:]:
                prediction = 'Yes'
            elif 'no' in resp[-20:]:
                prediction = 'No'
            else:
                prediction = 'Unknown'

            gt = 'Yes' if answer.lower() in ['yes', 'true'] else 'No'

            return {
                'id': sample.get('id'),
                'logic': sample.get('logic'),
                'depth': sample.get('depth'),
                'question': question,
                'ground_truth': gt,
                'prediction': prediction,
                'correct': prediction == gt,
                'mode': mode
            }
        except Exception as e:
            return {'error': str(e)}


async def main():
    parser = argparse.ArgumentParser(description='Test memorization by omitting premises')
    parser.add_argument('--api_key', required=True)
    parser.add_argument('--model', default='gpt-4o')
    parser.add_argument('--dataset', choices=['folio', 'multilogieval', 'both'], default='both')
    parser.add_argument('--folio_file', default='data/folio_original/folio-validation.json')
    parser.add_argument('--multi_dir', default='data/multi_logi_original/data')
    parser.add_argument('--mode', default='keep_none',
                        choices=['keep_none', 'keep_one', 'conclusion_only', 'full'],
                        help='Omission mode: keep_none removes all, keep_one keeps 1 premise, conclusion_only uses no context')
    parser.add_argument('--num_samples', type=int, default=50, help='Samples per dataset (0=all)')
    parser.add_argument('--concurrency', type=int, default=10)

    args = parser.parse_args()

    client = AsyncOpenAI(api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'results/memorization_omission/{args.mode}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    all_results = {'config': vars(args), 'results': {}}

    print("=" * 70)
    print("MEMORIZATION TEST VIA PREMISE OMISSION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Expected baseline accuracy:")
    print(f"  - FOLIO (3-class random): 33.3%")
    print(f"  - Multi-LogiEval (2-class random): 50%")
    print("=" * 70)

    # Test FOLIO
    if args.dataset in ['folio', 'both']:
        print(f"\n--- FOLIO ({args.mode}) ---")

        folio = load_folio(args.folio_file)
        if args.num_samples > 0:
            folio = random.sample(folio, min(args.num_samples, len(folio)))

        print(f"Testing {len(folio)} questions...")

        tasks = [test_folio_question(client, q, args.mode, args.model, semaphore) for q in folio]
        results = await tqdm_asyncio.gather(*tasks, desc="FOLIO")

        valid = [r for r in results if 'error' not in r]
        correct = sum(1 for r in valid if r['correct'])
        accuracy = correct / len(valid) if valid else 0

        print(f"\nFOLIO Results ({args.mode}):")
        print(f"  Accuracy: {correct}/{len(valid)} = {accuracy:.2%}")
        print(f"  Random baseline: 33.3%")

        if accuracy > 0.5 and args.mode in ['keep_none', 'conclusion_only']:
            print(f"  *** SUSPICIOUS: {accuracy:.1%} accuracy with NO premises! ***")
            print(f"  *** This strongly suggests memorization! ***")

        # Breakdown by label
        by_label = {}
        for r in valid:
            gt = r['ground_truth']
            by_label.setdefault(gt, {'correct': 0, 'total': 0})
            by_label[gt]['total'] += 1
            if r['correct']:
                by_label[gt]['correct'] += 1

        print("\n  By ground truth label:")
        for gt, stats in sorted(by_label.items()):
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"    {gt}: {stats['correct']}/{stats['total']} = {acc:.2%}")

        all_results['results']['folio'] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(valid),
            'by_label': by_label,
            'details': results
        }

    # Test Multi-LogiEval
    if args.dataset in ['multilogieval', 'both']:
        print(f"\n--- Multi-LogiEval ({args.mode}) ---")

        multi = load_multilogieval(args.multi_dir, max_per_file=3)
        if args.num_samples > 0:
            multi = random.sample(multi, min(args.num_samples, len(multi)))

        print(f"Testing {len(multi)} questions...")

        tasks = [test_multilogi_question(client, s, args.mode, args.model, semaphore) for s in multi]
        results = await tqdm_asyncio.gather(*tasks, desc="Multi-LogiEval")

        valid = [r for r in results if 'error' not in r]
        correct = sum(1 for r in valid if r['correct'])
        accuracy = correct / len(valid) if valid else 0

        print(f"\nMulti-LogiEval Results ({args.mode}):")
        print(f"  Accuracy: {correct}/{len(valid)} = {accuracy:.2%}")
        print(f"  Random baseline: 50%")

        if accuracy > 0.7 and args.mode in ['keep_none', 'conclusion_only']:
            print(f"  *** SUSPICIOUS: {accuracy:.1%} accuracy with NO context! ***")
            print(f"  *** This strongly suggests memorization! ***")

        all_results['results']['multilogieval'] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(valid),
            'details': results
        }

    # Save
    output_file = os.path.join(output_dir, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # Final summary
    print("\n" + "=" * 70)
    print("MEMORIZATION TEST SUMMARY")
    print("=" * 70)

    if 'folio' in all_results['results']:
        f = all_results['results']['folio']
        print(f"FOLIO:         {f['accuracy']:.2%} (baseline: 33.3%)")
        if f['accuracy'] > 0.5 and args.mode != 'full':
            print("               ^^^ POSSIBLE MEMORIZATION ^^^")

    if 'multilogieval' in all_results['results']:
        m = all_results['results']['multilogieval']
        print(f"Multi-LogiEval: {m['accuracy']:.2%} (baseline: 50%)")
        if m['accuracy'] > 0.7 and args.mode != 'full':
            print("               ^^^ POSSIBLE MEMORIZATION ^^^")


if __name__ == '__main__':
    asyncio.run(main())
