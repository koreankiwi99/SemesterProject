#!/usr/bin/env python3
"""
Rerun failed examples from result files and update them in place.
"""

import argparse
import asyncio
import json
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# Add src to path
import sys
sys.path.insert(0, 'src')

from utils.answer_parsing import parse_folio_answer, parse_multilogieval_answer
from datasets.folio import load_folio
from datasets.multilogieval import load_and_sample_multilogieval


async def run_folio_cot(story_data, client, model, system_prompt, user_template):
    """Run FOLIO CoT for a single story."""
    premises = story_data['premises']
    questions = story_data['questions']

    # Build question text
    q_text = ""
    for i, q in enumerate(questions, 1):
        q_text += f"\nQuestion {i}: Based on the above information, is the following statement true, false, or uncertain? {q['conclusion']}\n"

    user_prompt = user_template.replace("{premises}", premises)
    user_prompt = user_prompt.replace("{questions}", q_text)
    user_prompt = user_prompt.replace("{num_questions}", str(len(questions)))

    messages = [
        {"role": "system", "content": system_prompt.replace("{num_questions}", str(len(questions)))},
        {"role": "user", "content": user_prompt}
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages
    )

    model_response = response.choices[0].message.content

    # Parse answers
    results = []
    for i, q in enumerate(questions):
        answer = parse_folio_answer(model_response, question_num=i+1)
        gt = q.get('label', 'Unknown')
        if gt == 'Uncertain':
            gt = 'Unknown'
        results.append({
            'example_id': q.get('example_id'),
            'conclusion': q['conclusion'],
            'ground_truth': gt,
            'prediction': answer,
            'correct': answer == gt
        })

    return {
        'story_id': story_data['story_id'],
        'premises': premises,
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'model': model,
        'model_config': {},
        'model_response': model_response,
        'results': results
    }


async def run_multilogieval_cot(sample, client, model, system_prompt, user_template):
    """Run Multi-LogiEval CoT for a single sample."""
    context = sample['context']
    question = sample['question']

    user_prompt = user_template.replace("{context}", context).replace("{question}", question)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages
    )

    model_response = response.choices[0].message.content
    prediction = parse_multilogieval_answer(model_response)
    gt = sample['answer'].strip().lower()
    gt = 'Yes' if gt in ['yes', 'true'] else 'No'

    return {
        'question_num': sample.get('question_num', 0),
        'logic_type': sample['logic_type'],
        'depth': sample['depth'],
        'depth_dir': sample.get('depth_dir', f"{sample['depth']}_Data"),
        'rule': sample['rule'],
        'context': context,
        'question': question,
        'ground_truth': gt,
        'prediction': prediction,
        'correct': prediction == gt,
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'model': model,
        'model_config': {},
        'full_response': model_response,
        'source_file': sample.get('source_file', '')
    }


async def rerun_folio_errors(result_file, folio_file, client, model):
    """Rerun failed FOLIO examples."""
    with open(result_file) as f:
        results = json.load(f)

    # Find error indices
    error_indices = [i for i, r in enumerate(results) if 'error' in r]
    if not error_indices:
        print(f"  No errors found in {result_file}")
        return 0

    print(f"  Found {len(error_indices)} errors at indices: {error_indices}")

    # Load FOLIO data
    folio_data = load_folio(folio_file)

    # Group by story
    from datasets.folio import group_folio_by_story
    stories = group_folio_by_story(folio_data)
    story_map = {s['story_id']: s for s in stories}

    # Load prompts
    with open('prompts/folio/cot_system.txt') as f:
        system_prompt = f.read()
    with open('prompts/folio/cot_user.txt') as f:
        user_template = f.read()

    # Rerun each error
    fixed = 0
    for idx in error_indices:
        story_id = results[idx].get('story_id')
        if story_id not in story_map:
            print(f"    Story {story_id} not found, skipping")
            continue

        story_data = story_map[story_id]
        print(f"    Rerunning story_id={story_id}...")

        try:
            new_result = await run_folio_cot(story_data, client, model, system_prompt, user_template)
            results[idx] = new_result
            fixed += 1
            print(f"      Success!")
        except Exception as e:
            print(f"      Failed again: {e}")

    # Save updated results
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    return fixed


async def rerun_multilogieval_errors(result_file, data_dir, client, model, is_simple=False):
    """Rerun failed Multi-LogiEval examples."""
    with open(result_file) as f:
        results = json.load(f)

    # Find error indices
    error_indices = [i for i, r in enumerate(results) if 'error' in r]
    if not error_indices:
        print(f"  No errors found in {result_file}")
        return 0

    print(f"  Found {len(error_indices)} errors at indices: {error_indices}")

    # Load prompts
    if is_simple:
        system_prompt = "You are a logical reasoning expert. Answer based only on the given context."
        user_template = "Context: {context}\n\nQuestion: {question}\n\nAnswer with exactly: Yes or No"
    else:
        with open('prompts/multilogi/zero_shot_cot_system.txt') as f:
            system_prompt = f.read()
        with open('prompts/multilogi/zero_shot_cot_user.txt') as f:
            user_template = f.read()

    # Load all Multi-LogiEval data to find matching samples
    all_samples = load_and_sample_multilogieval(data_dir, samples_per_combination=100, seed=42)

    # Create lookup
    sample_map = {}
    for s in all_samples:
        key = (s['logic_type'], s['depth'], s['rule'], s['context'][:50])
        sample_map[key] = s

    # Rerun each error
    fixed = 0
    for idx in error_indices:
        r = results[idx]
        logic = r.get('logic_type')
        depth = r.get('depth')
        rule = r.get('rule')
        context_prefix = r.get('context', '')[:50]

        key = (logic, depth, rule, context_prefix)
        if key not in sample_map:
            # Try to reconstruct from the error entry
            print(f"    Sample not found for {logic}/{depth}/{rule}, using stored data...")
            sample = {
                'logic_type': logic,
                'depth': depth,
                'rule': rule,
                'context': r.get('context', ''),
                'question': r.get('question', ''),
                'answer': r.get('ground_truth', 'Yes'),
                'question_num': r.get('question_num', idx),
                'source_file': r.get('source_file', '')
            }
        else:
            sample = sample_map[key]

        print(f"    Rerunning {logic}/{depth}/{rule}...")

        try:
            new_result = await run_multilogieval_cot(sample, client, model, system_prompt, user_template)
            new_result['question_num'] = r.get('question_num', idx)
            results[idx] = new_result
            fixed += 1
            print(f"      Success!")
        except Exception as e:
            print(f"      Failed again: {e}")

    # Save updated results
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    return fixed


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openrouter_key', help='OpenRouter API key')
    parser.add_argument('--openai_key', help='OpenAI API key')
    args = parser.parse_args()

    # Files to fix
    files_to_fix = [
        {
            'path': 'results/cot/folio/deepseek_r1/all_results.json',
            'type': 'folio',
            'model': 'deepseek/deepseek-r1-0528',
            'provider': 'openrouter'
        },
        {
            'path': 'results/cot/folio/mistral_large/all_results.json',
            'type': 'folio',
            'model': 'mistralai/mistral-large-2411',
            'provider': 'openrouter'
        },
        {
            'path': 'results/cot/multilogieval/deepseek_r1/all_results.json',
            'type': 'multilogieval',
            'model': 'deepseek/deepseek-r1-0528',
            'provider': 'openrouter'
        },
        {
            'path': 'results/archive/multilogieval_simple_20251205_221137/all_results.json',
            'type': 'multilogieval_simple',
            'model': 'gpt-5',
            'provider': 'openai'
        },
    ]

    total_fixed = 0

    for file_info in files_to_fix:
        print(f"\n{'='*60}")
        print(f"Processing: {file_info['path']}")
        print(f"Model: {file_info['model']}")
        print(f"{'='*60}")

        # Create appropriate client
        if file_info['provider'] == 'openrouter':
            if not args.openrouter_key:
                print("  Skipping - no OpenRouter key provided")
                continue
            client = AsyncOpenAI(
                api_key=args.openrouter_key,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            if not args.openai_key:
                print("  Skipping - no OpenAI key provided")
                continue
            client = AsyncOpenAI(api_key=args.openai_key)

        try:
            if file_info['type'] == 'folio':
                fixed = await rerun_folio_errors(
                    file_info['path'],
                    'data/folio_original/folio-validation.json',
                    client,
                    file_info['model']
                )
            elif file_info['type'] == 'multilogieval':
                fixed = await rerun_multilogieval_errors(
                    file_info['path'],
                    'data/multi_logi_original/data',
                    client,
                    file_info['model'],
                    is_simple=False
                )
            elif file_info['type'] == 'multilogieval_simple':
                fixed = await rerun_multilogieval_errors(
                    file_info['path'],
                    'data/multi_logi_original/data',
                    client,
                    file_info['model'],
                    is_simple=True
                )

            total_fixed += fixed
            print(f"  Fixed {fixed} errors")
        except Exception as e:
            print(f"  Error processing file: {e}")

    print(f"\n{'='*60}")
    print(f"Total fixed: {total_fixed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
