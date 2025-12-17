"""
Add adversarial distractors to Multi-LogiEval problems.

Strategy: Use LLM to generate distractors that look relevant but don't
change the logical answer. Since Multi-LogiEval has formal ground truth
(generated from rules), we only need to ensure distractors don't
syntactically match rule variables.

Pipeline:
1. Load Multi-LogiEval problem
2. LLM generates k distractor sentences (same domain, different entities/relations)
3. Insert distractors into context
4. Ground truth stays the same (rules unchanged)
"""

import json
import asyncio
import argparse
from pathlib import Path
from openai import AsyncOpenAI
import random


DISTRACTOR_PROMPT = """You are helping create harder logical reasoning problems by adding distracting information.

Given this logical reasoning problem:

CONTEXT:
{context}

QUESTION:
{question}

ANSWER: {answer}

Generate {k} distractor sentences that:
1. Are about the SAME people/entities mentioned in the context
2. Sound relevant and plausible
3. Do NOT affect the logical answer (don't add new implications or contradictions)
4. Use different predicates/relations than the key logical statements

Good distractors: Background info, personality traits, unrelated activities, physical descriptions
Bad distractors: New if-then rules, contradictions, statements that could change the inference

Return ONLY a JSON array of {k} strings, nothing else.
Example: ["Ashley also enjoys cooking.", "The weather was sunny that day."]
"""


async def generate_distractors(client, context: str, question: str, answer: str, k: int = 3, model: str = "gpt-4o-mini") -> list[str]:
    """Generate k distractor sentences using LLM."""
    prompt = DISTRACTOR_PROMPT.format(
        context=context,
        question=question,
        answer=answer,
        k=k
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()

        # Try to extract JSON array from response
        # Handle cases where LLM adds extra text
        import re
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            content = json_match.group()

        # Parse JSON array
        distractors = json.loads(content)
        if isinstance(distractors, list):
            # Clean up each distractor
            distractors = [d.strip().rstrip('.') + '.' for d in distractors if isinstance(d, str)]
            if len(distractors) >= k:
                return distractors[:k]
            else:
                print(f"Warning: Expected {k} distractors, got {len(distractors)}")
                return distractors
        else:
            print(f"Warning: Got non-list response: {type(distractors)}")
            return []
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"  Content was: {content[:200]}...")
        return []
    except Exception as e:
        print(f"Error generating distractors: {e}")
        return []


def insert_distractors(context: str, distractors: list[str], strategy: str = "random") -> str:
    """Insert distractors into the context.

    Strategies:
    - random: Insert at random positions between sentences
    - beginning: Add all at the beginning
    - end: Add all at the end
    - interleaved: Spread evenly throughout
    """
    sentences = context.split('. ')
    sentences = [s.strip() for s in sentences if s.strip()]

    if strategy == "beginning":
        result = distractors + sentences
    elif strategy == "end":
        result = sentences + distractors
    elif strategy == "interleaved":
        result = []
        distractor_idx = 0
        step = max(1, len(sentences) // (len(distractors) + 1))
        for i, sent in enumerate(sentences):
            result.append(sent)
            if distractor_idx < len(distractors) and (i + 1) % step == 0:
                result.append(distractors[distractor_idx])
                distractor_idx += 1
        # Add remaining distractors
        result.extend(distractors[distractor_idx:])
    else:  # random
        result = sentences.copy()
        for d in distractors:
            pos = random.randint(0, len(result))
            result.insert(pos, d)

    # Reconstruct context
    return '. '.join(result) + ('.' if not result[-1].endswith('.') else '')


async def process_file(client, input_path: Path, output_path: Path, k: int = 3, model: str = "gpt-4o-mini"):
    """Process a single Multi-LogiEval file."""
    with open(input_path, encoding='utf-8', errors='ignore') as f:
        data = json.load(f)

    new_samples = []
    for sample in data['samples']:
        distractors = await generate_distractors(
            client,
            sample['context'],
            sample['question'],
            sample['answer'],
            k=k,
            model=model
        )

        if distractors:
            new_context = insert_distractors(sample['context'], distractors, strategy="random")
            new_sample = {
                **sample,
                'original_context': sample['context'],
                'context': new_context,
                'distractors': distractors
            }
        else:
            new_sample = {**sample, 'distractors': []}

        new_samples.append(new_sample)

    output_data = {
        **data,
        'samples': new_samples,
        'hardening': {
            'type': 'adversarial_distractors',
            'k': k,
            'model': model
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    return len(new_samples)


async def main():
    parser = argparse.ArgumentParser(description='Add adversarial distractors to Multi-LogiEval')
    parser.add_argument('--input_dir', type=str, default='data/multi_logi_original/data',
                        help='Input Multi-LogiEval data directory')
    parser.add_argument('--output_dir', type=str, default='data/multi_logi_adversarial',
                        help='Output directory for hardened data')
    parser.add_argument('--k', type=int, default=3, help='Number of distractors per problem')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model for distractor generation')
    parser.add_argument('--depths', type=str, nargs='+', default=['d3_Data', 'd5_Data'],
                        help='Which depths to process')
    parser.add_argument('--logic_types', type=str, nargs='+', default=['pl'],
                        help='Which logic types to process')
    parser.add_argument('--max_files', type=int, default=None, help='Max files to process (for testing)')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API key')

    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key required (--api_key or OPENAI_API_KEY env)")

    client = AsyncOpenAI(api_key=api_key)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    total_processed = 0
    files_processed = 0

    for depth in args.depths:
        for logic in args.logic_types:
            logic_dir = input_dir / depth / logic
            if not logic_dir.exists():
                print(f"Skipping {logic_dir} (not found)")
                continue

            for json_file in logic_dir.glob('*.json'):
                if args.max_files and files_processed >= args.max_files:
                    break

                output_path = output_dir / f"k{args.k}" / depth / logic / json_file.name
                print(f"Processing: {json_file} -> {output_path}")

                count = await process_file(client, json_file, output_path, k=args.k, model=args.model)
                total_processed += count
                files_processed += 1
                print(f"  Processed {count} samples")

    print(f"\nDone! Processed {total_processed} samples from {files_processed} files")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    asyncio.run(main())
