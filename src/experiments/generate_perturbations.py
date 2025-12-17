#!/usr/bin/env python3
"""
Generate perturbations for memorization testing.

This script generates perturbations using a separate model (e.g., gpt-4o-mini),
saves them to a file, and then they can be tested on the target model (e.g., gpt-5).

This separation ensures the perturbation generation doesn't bias the test results.
"""

import argparse
import asyncio
import json
import os
import random
import re
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

from datasets.folio import load_folio
from datasets.multilogieval import load_and_sample_multilogieval


async def generate_negation(client, model, statement, semaphore):
    """Use LLM to generate a proper logical negation."""
    async with semaphore:
        prompt = f"""Generate the logical negation of this statement.
The negation should directly contradict the original statement.
Output ONLY the negated statement, nothing else.

Original: {statement}

Negation:"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a logic expert. Generate clear logical negations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"It is NOT the case that {statement}"


async def find_critical_premise(client, model, premises_list, conclusion, semaphore):
    """Use LLM to identify the most critical premise."""
    async with semaphore:
        premises_numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(premises_list))
        prompt = f"""Given these premises and conclusion, identify which premise is MOST CRITICAL for deriving the conclusion.
If that premise were removed, the conclusion could no longer be derived.

Premises:
{premises_numbered}

Conclusion: {conclusion}

Output ONLY the number (1, 2, 3, etc.) of the most critical premise. Just the number."""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a logic expert. Identify critical premises."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0
            )
            idx_str = response.choices[0].message.content.strip()
            idx = int(re.search(r'\d+', idx_str).group()) - 1
            if 0 <= idx < len(premises_list):
                return idx, premises_list[idx]
        except Exception:
            pass

        # Fallback: use word overlap heuristic
        conclusion_words = set(re.findall(r'\b\w+\b', conclusion.lower()))
        best_idx, best_score = 0, 0
        for i, p in enumerate(premises_list):
            p_words = set(re.findall(r'\b\w+\b', p.lower()))
            score = len(p_words & conclusion_words)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx, premises_list[best_idx]


async def generate_folio_perturbation(example, client, model, semaphore):
    """Generate perturbations for a FOLIO example."""
    premises_str = example.get('premises', '')
    premises_list = [p.strip() for p in premises_str.split('\n') if p.strip()]
    conclusion = example.get('conclusion', '')

    if len(premises_list) < 2:
        return None

    # Generate negation of conclusion
    negation = await generate_negation(client, model, conclusion, semaphore)

    # Find critical premise to remove
    critical_idx, critical_premise = await find_critical_premise(
        client, model, premises_list, conclusion, semaphore
    )

    # Create perturbed versions
    premises_with_negation = premises_list + [negation]
    premises_without_critical = premises_list[:critical_idx] + premises_list[critical_idx+1:]

    return {
        'example_id': example.get('example_id'),
        'story_id': example.get('story_id'),
        'original_premises': premises_list,
        'conclusion': conclusion,
        'ground_truth': example.get('label'),
        'perturbations': {
            'add_negation': {
                'negation_statement': negation,
                'perturbed_premises': premises_with_negation,
                'expected_answer_change': True,
                'description': f"Added negation: '{negation}'"
            },
            'remove_premise': {
                'removed_premise': critical_premise,
                'removed_index': critical_idx,
                'perturbed_premises': premises_without_critical,
                'expected_answer_change': True,
                'description': f"Removed premise {critical_idx+1}: '{critical_premise[:50]}...'"
            }
        }
    }


async def generate_multilogi_perturbation(sample, client, model, semaphore):
    """Generate perturbations for a Multi-LogiEval example."""
    context = sample.get('context', '')
    question = sample.get('question', '')
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if s.strip()]

    if len(sentences) < 2:
        return None

    # Generate negation based on the question
    negation = await generate_negation(
        client, model,
        f"The answer to '{question}' is Yes",
        semaphore
    )

    # Find critical sentence
    critical_idx, critical_sentence = await find_critical_premise(
        client, model, sentences, question, semaphore
    )

    # Create perturbed versions
    context_with_negation = context + " " + negation
    sentences_without_critical = sentences[:critical_idx] + sentences[critical_idx+1:]
    context_without_critical = ' '.join(sentences_without_critical)

    return {
        'logic_type': sample.get('logic_type'),
        'depth': sample.get('depth'),
        'rule': sample.get('rule'),
        'original_context': context,
        'question': question,
        'ground_truth': sample.get('answer'),
        'perturbations': {
            'add_negation': {
                'negation_statement': negation,
                'perturbed_context': context_with_negation,
                'expected_answer_change': True,
                'description': f"Added negation: '{negation}'"
            },
            'remove_premise': {
                'removed_sentence': critical_sentence,
                'removed_index': critical_idx,
                'perturbed_context': context_without_critical,
                'expected_answer_change': True,
                'description': f"Removed sentence {critical_idx+1}: '{critical_sentence[:50]}...'"
            }
        }
    }


async def main():
    parser = argparse.ArgumentParser(description='Generate perturbations for memorization testing')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--generator_model', default='gpt-4o-mini',
                        help='Model to use for generating perturbations (default: gpt-4o-mini)')
    parser.add_argument('--dataset', choices=['folio', 'multilogieval', 'both'], default='both')
    parser.add_argument('--folio_file', default='data/folio_original/folio-validation.json')
    parser.add_argument('--multi_dir', default='data/multi_logi_original/data')
    parser.add_argument('--multi_samples_per_combo', type=int, default=10,
                        help='Samples per (logic_type, depth) combination for Multi-LogiEval')
    parser.add_argument('--multi_d5_only', action='store_true',
                        help='Only use depth-5 data for Multi-LogiEval')
    parser.add_argument('--num_samples', type=int, default=100, help='Samples per dataset (for FOLIO)')
    parser.add_argument('--concurrency', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='data/generated/perturbations')

    args = parser.parse_args()
    random.seed(args.seed)

    client = AsyncOpenAI(api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PERTURBATION GENERATION")
    print("=" * 70)
    print(f"Generator Model: {args.generator_model}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    all_perturbations = {}

    # Generate FOLIO perturbations
    if args.dataset in ['folio', 'both']:
        folio_data = load_folio(args.folio_file)
        # Only True/False examples (Unknown can't show memorization)
        folio_data = [ex for ex in folio_data if ex.get('label') in ['True', 'False']]

        if args.num_samples > 0 and args.num_samples < len(folio_data):
            folio_data = random.sample(folio_data, args.num_samples)

        print(f"\nGenerating FOLIO perturbations: {len(folio_data)} examples")
        tasks = [generate_folio_perturbation(ex, client, args.generator_model, semaphore)
                 for ex in folio_data]
        results = await tqdm_asyncio.gather(*tasks, desc="FOLIO")
        all_perturbations['folio'] = [r for r in results if r is not None]
        print(f"Generated {len(all_perturbations['folio'])} FOLIO perturbations")

    # Generate Multi-LogiEval perturbations
    if args.dataset in ['multilogieval', 'both']:
        if args.multi_d5_only:
            # Use d5_only data
            d5_dir = 'data/multi_logi_d5_only'
            multi_data = load_and_sample_multilogieval(
                d5_dir,
                samples_per_combination=args.multi_samples_per_combo,
                seed=args.seed
            )
            print(f"Using d5_only data from {d5_dir}")
        else:
            multi_data = load_and_sample_multilogieval(
                args.multi_dir,
                samples_per_combination=args.multi_samples_per_combo,
                seed=args.seed
            )

        print(f"\nGenerating Multi-LogiEval perturbations: {len(multi_data)} examples")
        tasks = [generate_multilogi_perturbation(s, client, args.generator_model, semaphore)
                 for s in multi_data]
        results = await tqdm_asyncio.gather(*tasks, desc="MultiLogiEval")
        all_perturbations['multilogieval'] = [r for r in results if r is not None]
        print(f"Generated {len(all_perturbations['multilogieval'])} Multi-LogiEval perturbations")

    # Save perturbations
    output_file = os.path.join(output_dir, f"perturbations_{args.generator_model}_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump({
            'generator_model': args.generator_model,
            'timestamp': timestamp,
            'config': vars(args),
            'perturbations': all_perturbations
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Perturbations saved to: {output_file}")
    print(f"{'=' * 70}")

    # Also save a summary for manual review
    summary_file = os.path.join(output_dir, f"perturbations_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Perturbation Summary\n")
        f.write(f"Generator: {args.generator_model}\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 70 + "\n\n")

        for dataset, perts in all_perturbations.items():
            f.write(f"\n{dataset.upper()} ({len(perts)} examples)\n")
            f.write("-" * 40 + "\n")

            # Show a few examples for manual verification
            for i, p in enumerate(perts[:5]):
                f.write(f"\nExample {i+1}:\n")
                if 'conclusion' in p:
                    f.write(f"  Conclusion: {p['conclusion'][:80]}...\n")
                else:
                    f.write(f"  Question: {p['question'][:80]}...\n")

                for pert_type, pert_data in p['perturbations'].items():
                    f.write(f"  {pert_type}: {pert_data['description']}\n")

    print(f"Summary for manual review: {summary_file}")

    return output_file


if __name__ == "__main__":
    asyncio.run(main())
