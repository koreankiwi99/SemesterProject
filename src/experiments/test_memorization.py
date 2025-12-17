#!/usr/bin/env python3
"""
Memorization test for FOLIO and Multi-LogiEval.

Tests if the model memorized the benchmark by omitting premises and checking
if it can still answer correctly (which would indicate memorization).

Based on test_folio_async.py and test_multi_async.py structure.
"""

import argparse
import asyncio
import random
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

from utils.answer_parsing import parse_folio_answer, parse_multilogieval_answer, normalize_answer
from utils.savers import MemorizationSaver
from datasets.folio import load_folio
from datasets.multilogieval import load_and_sample_multilogieval


def omit_premises_folio(premises_str: str, mode: str, keep_n: int = 1) -> str:
    """Omit premises from FOLIO (newline-separated)."""
    if mode == 'full':
        return premises_str

    sentences = [s.strip() for s in premises_str.split('\n') if s.strip()]

    if mode == 'keep_none':
        return "[No premises provided]"
    elif mode == 'keep_first':
        return '\n'.join(sentences[:keep_n]) if sentences else "[No premises]"
    elif mode == 'keep_last':
        return '\n'.join(sentences[-keep_n:]) if sentences else "[No premises]"
    elif mode == 'keep_random':
        if len(sentences) <= keep_n:
            return premises_str
        kept = random.sample(sentences, keep_n)
        return '\n'.join(kept)

    return premises_str


def omit_context_multilogi(context: str, mode: str, keep_n: int = 1) -> str:
    """Omit context from Multi-LogiEval (period-separated)."""
    import re

    if mode == 'full':
        return context

    sentences = [s.strip() + '.' for s in re.split(r'\.(?:\s|$)', context) if s.strip()]

    if mode == 'keep_none':
        return "[No context provided]"
    elif mode == 'keep_first':
        return ' '.join(sentences[:keep_n]) if sentences else "[No context]"
    elif mode == 'keep_last':
        return ' '.join(sentences[-keep_n:]) if sentences else "[No context]"
    elif mode == 'keep_random':
        if len(sentences) <= keep_n:
            return context
        kept = random.sample(sentences, keep_n)
        return ' '.join(kept)

    return context


async def test_folio_question(example, client, model, mode, keep_n, semaphore):
    """Test a single FOLIO question with premise omission."""
    async with semaphore:
        original_premises = example['premises']
        modified_premises = omit_premises_folio(original_premises, mode, keep_n)

        system_prompt = """You are a logical reasoning assistant. Determine if conclusions follow from premises.
Answer with exactly one of: True, False, or Unknown"""

        user_prompt = f"""Premises:
{modified_premises}

Conclusion: {example['conclusion']}

Based on the premises, is the conclusion True, False, or Unknown?"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            answer = response.choices[0].message.content.strip()
            prediction = parse_folio_answer(answer)
            ground_truth = normalize_answer(example['label'], answer_format="true_false")

            return {
                'example_id': example.get('example_id'),
                'story_id': example.get('story_id'),
                'conclusion': example['conclusion'],
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': prediction == ground_truth,
                'mode': mode,
                'keep_n': keep_n,
                'num_original_premises': len([s for s in original_premises.split('\n') if s.strip()]),
                'modified_premises': modified_premises[:200] + '...' if len(modified_premises) > 200 else modified_premises
            }
        except Exception as e:
            return {
                'example_id': example.get('example_id'),
                'error': str(e)
            }


async def test_multilogi_question(sample, client, model, mode, keep_n, semaphore):
    """Test a single Multi-LogiEval question with context omission."""
    async with semaphore:
        original_context = sample['context']
        modified_context = omit_context_multilogi(original_context, mode, keep_n)

        system_prompt = """You are a logical reasoning expert. Answer based only on the given context.
Answer with exactly: Yes or No"""

        user_prompt = f"""Context: {modified_context}

Question: {sample['question']}

Answer:"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            answer = response.choices[0].message.content.strip()
            prediction = parse_multilogieval_answer(answer)
            ground_truth = normalize_answer(sample['answer'], answer_format="yes_no")

            return {
                'logic_type': sample.get('logic_type'),
                'depth': sample.get('depth'),
                'rule': sample.get('rule'),
                'question': sample['question'],
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': prediction == ground_truth,
                'mode': mode,
                'keep_n': keep_n
            }
        except Exception as e:
            return {
                'logic_type': sample.get('logic_type'),
                'error': str(e)
            }


async def main_async():
    parser = argparse.ArgumentParser(description='Memorization test via premise omission')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--model', default='gpt-4o', help='Model to test')
    parser.add_argument('--dataset', choices=['folio', 'multilogieval', 'both'], default='both')
    parser.add_argument('--folio_file', default='data/folio_original/folio-validation.json')
    parser.add_argument('--multi_dir', default='data/multi_logi_original/data')
    parser.add_argument('--mode', default='keep_none',
                        choices=['keep_none', 'keep_first', 'keep_last', 'keep_random', 'full'])
    parser.add_argument('--keep_n', type=int, default=1, help='Number of premises to keep')
    parser.add_argument('--num_samples', type=int, default=100, help='Samples per dataset (0=all)')
    parser.add_argument('--concurrency', type=int, default=10)
    parser.add_argument('--output_dir', default='results/memorization_test')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    client = AsyncOpenAI(api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    # Initialize saver
    saver = MemorizationSaver(args.output_dir, f"{args.mode}_k{args.keep_n}")

    print("=" * 70)
    print("MEMORIZATION TEST VIA PREMISE OMISSION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}, Keep N: {args.keep_n}")
    print(f"Expected baselines: FOLIO ~33.3%, Multi-LogiEval ~50%")
    print("=" * 70)

    all_results = {'folio': [], 'multilogieval': []}

    # Test FOLIO
    if args.dataset in ['folio', 'both']:
        print(f"\n--- Testing FOLIO ({args.mode}) ---")
        folio_data = load_folio(args.folio_file)

        if args.num_samples > 0 and args.num_samples < len(folio_data):
            folio_data = random.sample(folio_data, args.num_samples)

        print(f"Testing {len(folio_data)} questions...")

        tasks = [
            test_folio_question(ex, client, args.model, args.mode, args.keep_n, semaphore)
            for ex in folio_data
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="FOLIO")
        all_results['folio'] = results

        valid = [r for r in results if 'error' not in r]
        correct = sum(1 for r in valid if r['correct'])
        accuracy = correct / len(valid) if valid else 0

        print(f"\nFOLIO Results:")
        print(f"  Accuracy: {correct}/{len(valid)} = {accuracy:.2%}")
        print(f"  Random baseline: 33.3%")

        if accuracy > 0.5 and args.mode == 'keep_none':
            print(f"  *** WARNING: High accuracy with NO premises suggests memorization! ***")

        # Breakdown by label
        by_label = {}
        for r in valid:
            gt = r['ground_truth']
            by_label.setdefault(gt, {'correct': 0, 'total': 0})
            by_label[gt]['total'] += 1
            if r['correct']:
                by_label[gt]['correct'] += 1

        print("\n  By ground truth:")
        for gt, stats in sorted(by_label.items()):
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"    {gt}: {stats['correct']}/{stats['total']} = {acc:.2%}")

    # Test Multi-LogiEval
    if args.dataset in ['multilogieval', 'both']:
        print(f"\n--- Testing Multi-LogiEval ({args.mode}) ---")

        multi_data = load_and_sample_multilogieval(
            args.multi_dir,
            samples_per_combination=max(5, args.num_samples // 15),
            seed=args.seed
        )

        if args.num_samples > 0 and args.num_samples < len(multi_data):
            multi_data = random.sample(multi_data, args.num_samples)

        print(f"Testing {len(multi_data)} questions...")

        tasks = [
            test_multilogi_question(s, client, args.model, args.mode, args.keep_n, semaphore)
            for s in multi_data
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Multi-LogiEval")
        all_results['multilogieval'] = results

        valid = [r for r in results if 'error' not in r]
        correct = sum(1 for r in valid if r['correct'])
        accuracy = correct / len(valid) if valid else 0

        print(f"\nMulti-LogiEval Results:")
        print(f"  Accuracy: {correct}/{len(valid)} = {accuracy:.2%}")
        print(f"  Random baseline: 50%")

        if accuracy > 0.7 and args.mode == 'keep_none':
            print(f"  *** WARNING: High accuracy with NO context suggests memorization! ***")

    # Save results
    saver.save_all(all_results, vars(args))

    print(f"\n{'='*70}")
    print("MEMORIZATION TEST SUMMARY")
    print("=" * 70)

    if all_results['folio']:
        valid = [r for r in all_results['folio'] if 'error' not in r]
        acc = sum(1 for r in valid if r['correct']) / len(valid) if valid else 0
        print(f"FOLIO:          {acc:.2%} (baseline: 33.3%)")

    if all_results['multilogieval']:
        valid = [r for r in all_results['multilogieval'] if 'error' not in r]
        acc = sum(1 for r in valid if r['correct']) / len(valid) if valid else 0
        print(f"Multi-LogiEval: {acc:.2%} (baseline: 50%)")


if __name__ == "__main__":
    asyncio.run(main_async())
