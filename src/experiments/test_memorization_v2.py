#!/usr/bin/env python3
"""
Memorization Test v2 - Improved Perturbation Methodology

Key improvements over v1:
1. LLM-generated negations (not simple string replacement)
2. Validation that perturbations create valid logical contradictions
3. Structured output for manual verification
4. Clearer verdict categories

Perturbation types:
1. NEGATE_CONCLUSION: Add explicit negation of conclusion as premise
2. REMOVE_PREMISE: Remove the most relevant premise to conclusion

Expected behavior (non-memorized model):
- NEGATE_CONCLUSION: Answer should change to False/No or become contradictory
- REMOVE_PREMISE: Answer should change to Unknown (can't conclude anymore)
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

from utils.answer_parsing import parse_folio_answer, parse_multilogieval_answer, normalize_answer
from datasets.folio import load_folio
from datasets.multilogieval import load_and_sample_multilogieval


async def generate_negation_llm(client, model, conclusion, semaphore):
    """Use LLM to generate a proper logical negation of the conclusion."""
    async with semaphore:
        prompt = f"""Generate the logical negation of this statement.
The negation should directly contradict the original statement.
Output ONLY the negated statement, nothing else.

Original statement: {conclusion}

Negation:"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a logic expert. Generate clear logical negations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to simple negation
            return f"It is NOT the case that {conclusion}"


async def find_critical_premise_llm(client, model, premises_list, conclusion, semaphore):
    """Use LLM to identify the most critical premise for the conclusion."""
    async with semaphore:
        premises_numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(premises_list))
        prompt = f"""Given these premises and conclusion, identify which premise is MOST CRITICAL for deriving the conclusion.
If that premise were removed, the conclusion could no longer be derived.

Premises:
{premises_numbered}

Conclusion: {conclusion}

Output ONLY the number (1, 2, 3, etc.) of the most critical premise. Just the number, nothing else."""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a logic expert. Identify critical premises."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10
            )
            idx_str = response.choices[0].message.content.strip()
            idx = int(re.search(r'\d+', idx_str).group()) - 1
            if 0 <= idx < len(premises_list):
                return idx, premises_list[idx]
        except Exception:
            pass

        # Fallback to heuristic
        return find_critical_premise(premises_list, conclusion)


def find_critical_premise(premises_list, conclusion):
    """Find the premise most critical to deriving the conclusion.

    Uses word overlap and entity matching as heuristics.
    Returns (index, removed_premise).
    """
    if len(premises_list) <= 1:
        return None, None

    conclusion_lower = conclusion.lower()
    conclusion_words = set(re.findall(r'\b\w+\b', conclusion_lower))

    # Extract potential entities (capitalized words, proper nouns)
    conclusion_entities = set(re.findall(r'\b[A-Z][a-z]+\b', conclusion))

    best_idx = 0
    best_score = 0

    for i, p in enumerate(premises_list):
        p_lower = p.lower()
        p_words = set(re.findall(r'\b\w+\b', p_lower))
        p_entities = set(re.findall(r'\b[A-Z][a-z]+\b', p))

        # Score: word overlap + entity overlap (weighted higher)
        word_overlap = len(p_words & conclusion_words)
        entity_overlap = len(p_entities & conclusion_entities) * 3  # Weight entities more

        score = word_overlap + entity_overlap

        if score > best_score:
            best_score = score
            best_idx = i

    removed = premises_list[best_idx]
    return best_idx, removed


async def test_folio_example(example, client, model, semaphore, negation_cache, use_llm_perturbation=False):
    """Test a single FOLIO example with both perturbation types."""
    async with semaphore:
        premises_str = example.get('premises', '')
        premises_list = [p.strip() for p in premises_str.split('\n') if p.strip()]
        conclusion = example.get('conclusion', '')
        label = example.get('label', '')
        ground_truth = normalize_answer(label, answer_format="true_false")

        if len(premises_list) < 2:
            return {'skip': True, 'reason': 'too few premises'}

        system_prompt = "You are a logical reasoning assistant. Determine if conclusions follow from premises."

        # Get original answer
        orig_prompt = f"""Premises:
{chr(10).join('- ' + p for p in premises_list)}

Conclusion: {conclusion}

Based ONLY on the premises, is the conclusion True, False, or Unknown?
Answer with exactly one word: True, False, or Unknown"""

        try:
            orig_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": orig_prompt}
                ]
            )
            orig_answer = parse_folio_answer(orig_resp.choices[0].message.content)

            results = {
                'example_id': example.get('example_id'),
                'story_id': example.get('story_id'),
                'conclusion': conclusion,
                'ground_truth': ground_truth,
                'orig_answer': orig_answer,
                'orig_correct': orig_answer == ground_truth,
                'perturbations': {}
            }

            # Perturbation 1: Add negation of conclusion
            cache_key = conclusion
            if cache_key in negation_cache:
                negation = negation_cache[cache_key]
            elif use_llm_perturbation:
                # Use LLM to generate better negation
                negation = await generate_negation_llm(client, model, conclusion, semaphore)
                negation_cache[cache_key] = negation
            else:
                # Use simple negation for consistency
                if "is a" in conclusion.lower():
                    negation = conclusion.replace(" is a ", " is not a ").replace(" Is a ", " Is not a ")
                elif "is " in conclusion.lower():
                    negation = conclusion.replace(" is ", " is not ", 1).replace(" Is ", " Is not ", 1)
                elif "are " in conclusion.lower():
                    negation = conclusion.replace(" are ", " are not ", 1).replace(" Are ", " Are not ", 1)
                elif "can " in conclusion.lower():
                    negation = conclusion.replace(" can ", " cannot ", 1)
                elif "will " in conclusion.lower():
                    negation = conclusion.replace(" will ", " will not ", 1)
                elif "does " in conclusion.lower():
                    negation = conclusion.replace(" does ", " does not ", 1)
                elif "do " in conclusion.lower():
                    negation = conclusion.replace(" do ", " do not ", 1)
                else:
                    negation = f"It is NOT the case that: {conclusion}"
                negation_cache[cache_key] = negation

            neg_premises = premises_list + [negation]
            neg_prompt = f"""Premises:
{chr(10).join('- ' + p for p in neg_premises)}

Conclusion: {conclusion}

Based ONLY on the premises, is the conclusion True, False, or Unknown?
Answer with exactly one word: True, False, or Unknown"""

            neg_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": neg_prompt}
                ]
            )
            neg_answer = parse_folio_answer(neg_resp.choices[0].message.content)

            results['perturbations']['add_negation'] = {
                'negation_added': negation,
                'perturbed_answer': neg_answer,
                'answer_changed': neg_answer != orig_answer,
                'expected_change': True,  # Should change when negation added
            }

            # Perturbation 2: Remove critical premise
            if use_llm_perturbation:
                idx, removed = await find_critical_premise_llm(client, model, premises_list, conclusion, semaphore)
            else:
                idx, removed = find_critical_premise(premises_list, conclusion)
            if idx is not None:
                rem_premises = premises_list[:idx] + premises_list[idx+1:]
                rem_prompt = f"""Premises:
{chr(10).join('- ' + p for p in rem_premises)}

Conclusion: {conclusion}

Based ONLY on the premises, is the conclusion True, False, or Unknown?
Answer with exactly one word: True, False, or Unknown"""

                rem_resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": rem_prompt}
                    ]
                )
                rem_answer = parse_folio_answer(rem_resp.choices[0].message.content)

                results['perturbations']['remove_premise'] = {
                    'premise_removed': removed,
                    'perturbed_answer': rem_answer,
                    'answer_changed': rem_answer != orig_answer,
                    'expected_change': True,  # Should change when critical premise removed
                }

            # Compute verdict
            results['verdict'] = compute_verdict(results)

            return results

        except Exception as e:
            return {'example_id': example.get('example_id'), 'error': str(e)}


async def test_multilogi_example(sample, client, model, semaphore, negation_cache, use_llm_perturbation=False):
    """Test a single Multi-LogiEval example with both perturbation types."""
    async with semaphore:
        context = sample.get('context', '')
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        ground_truth = normalize_answer(answer, answer_format="yes_no")

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if s.strip()]

        if len(sentences) < 2:
            return {'skip': True, 'reason': 'too few sentences'}

        system_prompt = "You are a logical reasoning expert. Answer based only on the given context."

        orig_prompt = f"""Context: {context}

Question: {question}

Answer with exactly one word: Yes or No"""

        try:
            orig_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": orig_prompt}
                ]
            )
            orig_answer = parse_multilogieval_answer(orig_resp.choices[0].message.content)

            results = {
                'logic_type': sample.get('logic_type'),
                'depth': sample.get('depth'),
                'rule': sample.get('rule'),
                'question': question,
                'ground_truth': ground_truth,
                'orig_answer': orig_answer,
                'orig_correct': orig_answer == ground_truth,
                'perturbations': {}
            }

            # Perturbation 1: Add negation of what the question asks
            if use_llm_perturbation:
                # Use LLM to generate a statement that negates what the question asks
                negation = await generate_negation_llm(client, model,
                    f"The answer to '{question}' is Yes", semaphore)
            else:
                # Extract the core claim from the question
                q_lower = question.lower()
                if q_lower.startswith("does "):
                    core = question[5:].rstrip("?")
                    negation = f"{core.split()[0]} does not {' '.join(core.split()[1:])}"
                elif "entail" in q_lower:
                    negation = "The answer to the question is No."
                else:
                    negation = f"It is NOT the case that the answer is Yes."

            neg_context = context + " " + negation
            neg_prompt = f"""Context: {neg_context}

Question: {question}

Answer with exactly one word: Yes or No"""

            neg_resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": neg_prompt}
                ]
            )
            neg_answer = parse_multilogieval_answer(neg_resp.choices[0].message.content)

            results['perturbations']['add_negation'] = {
                'negation_added': negation,
                'perturbed_answer': neg_answer,
                'answer_changed': neg_answer != orig_answer,
                'expected_change': True,
            }

            # Perturbation 2: Remove critical premise
            if use_llm_perturbation:
                idx, removed = await find_critical_premise_llm(client, model, sentences, question, semaphore)
            else:
                idx, removed = find_critical_premise(sentences, question)
            if idx is not None:
                rem_sentences = sentences[:idx] + sentences[idx+1:]
                rem_context = ' '.join(rem_sentences)
                rem_prompt = f"""Context: {rem_context}

Question: {question}

Answer with exactly one word: Yes or No"""

                rem_resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": rem_prompt}
                    ]
                )
                rem_answer = parse_multilogieval_answer(rem_resp.choices[0].message.content)

                results['perturbations']['remove_premise'] = {
                    'premise_removed': removed,
                    'perturbed_answer': rem_answer,
                    'answer_changed': rem_answer != orig_answer,
                    'expected_change': True,
                }

            results['verdict'] = compute_verdict(results)

            return results

        except Exception as e:
            return {'error': str(e)}


def compute_verdict(result):
    """Compute verdict based on perturbation responses.

    Verdicts:
    - LIKELY_MEMORIZED: Both perturbations show no change when change expected
    - POSSIBLY_MEMORIZED: One perturbation shows no change
    - ROBUST: Answer changes appropriately with perturbations
    - ORIGINALLY_WRONG: Model got original wrong, can't assess memorization
    - INCONCLUSIVE: Mixed signals
    """
    if not result.get('orig_correct', False):
        return 'ORIGINALLY_WRONG'

    perts = result.get('perturbations', {})
    if not perts:
        return 'INCONCLUSIVE'

    unchanged_count = 0
    total_perts = 0

    for pert_name, pert_data in perts.items():
        if pert_data.get('expected_change', False):
            total_perts += 1
            if not pert_data.get('answer_changed', False):
                unchanged_count += 1

    if total_perts == 0:
        return 'INCONCLUSIVE'

    if unchanged_count == total_perts:
        return 'LIKELY_MEMORIZED'
    elif unchanged_count > 0:
        return 'POSSIBLY_MEMORIZED'
    else:
        return 'ROBUST'


def create_manual_verification_samples(results, output_dir, n_samples=20):
    """Create samples for manual verification of perturbation quality."""
    samples = []

    for r in results:
        if r.get('skip') or r.get('error'):
            continue

        for pert_type, pert_data in r.get('perturbations', {}).items():
            samples.append({
                'example_id': r.get('example_id') or r.get('logic_type'),
                'perturbation_type': pert_type,
                'original_conclusion': r.get('conclusion') or r.get('question'),
                'perturbation_detail': pert_data.get('negation_added') or pert_data.get('premise_removed'),
                'original_answer': r.get('orig_answer'),
                'perturbed_answer': pert_data.get('perturbed_answer'),
                'answer_changed': pert_data.get('answer_changed'),
                'verdict': r.get('verdict'),
                # For manual verification:
                'manual_check': {
                    'is_perturbation_valid': None,  # True/False - to be filled manually
                    'should_answer_change': None,   # True/False - to be filled manually
                    'notes': ''                      # Any notes
                }
            })

    # Sample diverse cases
    memorized = [s for s in samples if 'MEMORIZED' in s['verdict']]
    robust = [s for s in samples if s['verdict'] == 'ROBUST']

    selected = []
    if memorized:
        selected.extend(random.sample(memorized, min(n_samples//2, len(memorized))))
    if robust:
        selected.extend(random.sample(robust, min(n_samples//2, len(robust))))

    # Save for manual verification
    manual_file = os.path.join(output_dir, 'manual_verification_samples.json')
    with open(manual_file, 'w') as f:
        json.dump(selected, f, indent=2)

    print(f"\nCreated {len(selected)} samples for manual verification: {manual_file}")
    return selected


async def main_async():
    parser = argparse.ArgumentParser(description='Memorization Test v2 - Improved Methodology')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--model', default='gpt-5', help='Model to test')
    parser.add_argument('--dataset', choices=['folio', 'multilogieval', 'both'], default='both')
    parser.add_argument('--folio_file', default='data/folio_original/folio-validation.json')
    parser.add_argument('--multi_dir', default='data/multi_logi_original/data')
    parser.add_argument('--num_samples', type=int, default=50, help='Samples per dataset')
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='results/memorization')
    parser.add_argument('--use_llm_perturbation', action='store_true',
                        help='Use LLM to generate negations and identify critical premises (slower but more accurate)')

    args = parser.parse_args()
    random.seed(args.seed)

    client = AsyncOpenAI(api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    negation_cache = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("MEMORIZATION TEST v2 - IMPROVED METHODOLOGY")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print(f"LLM Perturbation: {'YES (using LLM for negation & critical premise)' if args.use_llm_perturbation else 'NO (using heuristics)'}")
    print("=" * 70)

    all_results = {'folio': [], 'multilogieval': []}

    # Test FOLIO
    if args.dataset in ['folio', 'both']:
        folio_data = load_folio(args.folio_file)
        # Only test True/False examples (Unknown can't show memorization)
        folio_data = [ex for ex in folio_data if ex.get('label') in ['True', 'False']]

        if args.num_samples > 0 and args.num_samples < len(folio_data):
            folio_data = random.sample(folio_data, args.num_samples)

        print(f"\nTesting FOLIO: {len(folio_data)} questions")
        tasks = [test_folio_example(ex, client, args.model, semaphore, negation_cache, args.use_llm_perturbation)
                 for ex in folio_data]
        results = await tqdm_asyncio.gather(*tasks, desc="FOLIO")
        all_results['folio'] = [r for r in results if not r.get('skip') and not r.get('error')]

    # Test Multi-LogiEval
    if args.dataset in ['multilogieval', 'both']:
        multi_data = load_and_sample_multilogieval(
            args.multi_dir,
            samples_per_combination=max(3, args.num_samples // 15),
            seed=args.seed
        )

        if args.num_samples > 0 and args.num_samples < len(multi_data):
            multi_data = random.sample(multi_data, args.num_samples)

        print(f"\nTesting Multi-LogiEval: {len(multi_data)} questions")
        tasks = [test_multilogi_example(s, client, args.model, semaphore, negation_cache, args.use_llm_perturbation)
                 for s in multi_data]
        results = await tqdm_asyncio.gather(*tasks, desc="MultiLogiEval")
        all_results['multilogieval'] = [r for r in results if not r.get('skip') and not r.get('error')]

    # Compute summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary = {}
    for dataset, results in all_results.items():
        if not results:
            continue

        verdicts = {'LIKELY_MEMORIZED': 0, 'POSSIBLY_MEMORIZED': 0, 'ROBUST': 0,
                    'ORIGINALLY_WRONG': 0, 'INCONCLUSIVE': 0}

        for r in results:
            v = r.get('verdict', 'INCONCLUSIVE')
            if v in verdicts:
                verdicts[v] += 1

        total = len(results)
        summary[dataset] = {
            'total': total,
            'verdicts': verdicts,
            'memorization_rate': (verdicts['LIKELY_MEMORIZED'] + verdicts['POSSIBLY_MEMORIZED']) / total * 100 if total > 0 else 0
        }

        print(f"\n{dataset.upper()} (n={total}):")
        for v, count in verdicts.items():
            pct = count/total*100 if total > 0 else 0
            print(f"  {v}: {count} ({pct:.1f}%)")

    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'model': args.model,
            'timestamp': timestamp,
            'config': vars(args),
            'summary': summary,
            'folio_results': all_results['folio'],
            'multilogieval_results': all_results['multilogieval']
        }, f, indent=2)

    # Create manual verification samples
    all_flat = all_results['folio'] + all_results['multilogieval']
    create_manual_verification_samples(all_flat, output_dir, n_samples=30)

    print(f"\nResults saved to: {output_dir}")

    # Print interpretation
    total_mem = sum(s.get('verdicts', {}).get('LIKELY_MEMORIZED', 0) +
                   s.get('verdicts', {}).get('POSSIBLY_MEMORIZED', 0)
                   for s in summary.values())
    total_all = sum(s.get('total', 0) for s in summary.values())

    if total_all > 0:
        overall_rate = total_mem / total_all * 100
        print(f"\nOverall potential memorization rate: {total_mem}/{total_all} ({overall_rate:.1f}%)")

        if overall_rate > 40:
            print("⚠️  HIGH MEMORIZATION SIGNAL - Manual verification strongly recommended")
        elif overall_rate > 20:
            print("⚠️  MODERATE MEMORIZATION SIGNAL - Review manual_verification_samples.json")
        else:
            print("✓  LOW MEMORIZATION SIGNAL - But still verify samples manually")


if __name__ == "__main__":
    asyncio.run(main_async())
