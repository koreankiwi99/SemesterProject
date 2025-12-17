#!/usr/bin/env python3
"""
Group FOLIO questions by story_id and test with customizable prompts.
Tests multiple questions per story in a single LLM call.
"""

import argparse
import json
import os

from utils.prompts import load_prompt
from utils.answer_parsing import parse_folio_multiple_answers, normalize_answer
from utils.savers import FOLIOSaver
from datasets.folio import load_and_group_folio, build_folio_prompt_grouped


def test_model_grouped(story_examples, api_key, system_prompt_template, user_prompt_template,
                       model="gpt-5", model_config=None):
    """Test model on grouped questions from a single story.

    Args:
        story_examples: List of FOLIO examples from the same story
        api_key: OpenAI API key
        system_prompt_template: System prompt template
        user_prompt_template: User prompt template
        model: Model name to use
        model_config: Optional model configuration dict (temperature, reasoning_level, etc.)

    Returns:
        dict: Results for this story including all questions
    """
    import openai
    openai.api_key = api_key

    system_msg, user_prompt = build_folio_prompt_grouped(story_examples, system_prompt_template, user_prompt_template)

    # Build API call parameters
    api_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
    }

    # Add model config if provided
    if model_config:
        api_params.update(model_config)

    try:
        response = openai.chat.completions.create(**api_params)

        model_response = response.choices[0].message.content.strip()
        answers = parse_folio_multiple_answers(model_response, len(story_examples))

        results = []
        for i, (example, prediction) in enumerate(zip(story_examples, answers)):
            correct = normalize_answer(prediction, answer_format="true_false") == normalize_answer(example["label"], answer_format="true_false")
            results.append({
                'question_num': i + 1,
                'example_id': example.get('example_id'),
                'conclusion': example['conclusion'],
                'ground_truth': example['label'],
                'prediction': prediction,
                'correct': correct
            })

        return {
            'story_id': story_examples[0].get('story_id'),
            'premises': story_examples[0]['premises'],
            'system_prompt': system_msg,
            'user_prompt': user_prompt,
            'model': model,
            'model_config': model_config or {},
            'model_response': model_response,
            'results': results,
            'story_accuracy': sum(r['correct'] for r in results) / len(results)
        }

    except Exception as e:
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Test models on grouped FOLIO questions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python test_folio.py \\
      --api_key YOUR_KEY \\
      --folio_file data/folio_original/folio-validation.json \\
      --system_prompt prompts/folio/cot_system.txt \\
      --user_prompt prompts/folio/cot_user.txt \\
      --prompt_name cot \\
      --num_stories 10
        """
    )
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--folio_file', required=True, help='FOLIO JSON file')
    parser.add_argument('--system_prompt', required=True, help='Path to system prompt file')
    parser.add_argument('--user_prompt', required=True, help='Path to user prompt template file')
    parser.add_argument('--prompt_name', default='test', help='Name for output files (e.g., "cot", "lean")')
    parser.add_argument('--num_stories', type=int, default=5,
                        help='Number of stories to test (0 or negative = all)')
    parser.add_argument('--output_dir', default='results', help='Directory to save responses')
    parser.add_argument('--model', default='gpt-5', help='Model to use')
    parser.add_argument('--resume', type=str, help='Path to existing results directory to resume from')

    # Model configuration options
    parser.add_argument('--temperature', type=float, help='Sampling temperature (0.0-2.0)')
    parser.add_argument('--reasoning_effort', type=str, choices=['low', 'medium', 'high'],
                        help='Reasoning effort (for models like o1/o3)')
    parser.add_argument('--max_tokens', type=int, help='Maximum tokens in response')
    parser.add_argument('--top_p', type=float, help='Nucleus sampling parameter')

    args = parser.parse_args()

    # Build model config from args
    model_config = {}
    if args.temperature is not None:
        model_config['temperature'] = args.temperature
    if args.reasoning_effort is not None:
        model_config['reasoning_effort'] = args.reasoning_effort
    if args.max_tokens is not None:
        model_config['max_tokens'] = args.max_tokens
    if args.top_p is not None:
        model_config['top_p'] = args.top_p

    # Load prompts
    print(f"Loading prompts...")
    system_prompt_template = load_prompt(args.system_prompt)
    user_prompt_template = load_prompt(args.user_prompt)
    print(f"✓ Loaded system prompt from {args.system_prompt}")
    print(f"✓ Loaded user prompt from {args.user_prompt}")

    grouped_data = load_and_group_folio(args.folio_file)

    # Select stories to test
    if args.num_stories > 0:
        story_ids = list(grouped_data.keys())[:args.num_stories]
    else:
        story_ids = list(grouped_data.keys())

    # Handle resume functionality
    processed_story_ids = set()
    if args.resume:
        all_results_file = os.path.join(args.resume, 'all_results.json')
        if os.path.exists(all_results_file):
            print(f"\nResuming from: {args.resume}")
            with open(all_results_file, 'r') as f:
                previous_results = json.load(f)
            processed_story_ids = {r['story_id'] for r in previous_results if 'error' not in r or r.get('story_id')}
            print(f"Found {len(processed_story_ids)} already processed stories")
            print(f"Skipping: {sorted(processed_story_ids)}")
        else:
            print(f"Warning: Resume directory exists but no all_results.json found: {all_results_file}")

    # Filter out already processed stories
    remaining_story_ids = [sid for sid in story_ids if sid not in processed_story_ids]

    print(f"\nTesting {len(remaining_story_ids)} stories with {args.model}")
    if processed_story_ids:
        print(f"(Skipping {len(processed_story_ids)} already completed stories)")

    # Initialize saver (will resume if --resume provided)
    if args.resume:
        saver = FOLIOSaver(args.output_dir, args.prompt_name, resume_dir=args.resume)
    else:
        saver = FOLIOSaver(args.output_dir, args.prompt_name)

    total_questions = 0
    total_correct = 0

    try:
        for i, story_id in enumerate(remaining_story_ids):
            story_examples = grouped_data[story_id]
            total_index = len(processed_story_ids) + i + 1
            total_stories = len(story_ids)
            print(f"\nStory {total_index}/{total_stories} (ID: {story_id}): {len(story_examples)} questions")

            result = test_model_grouped(story_examples, args.api_key,
                                       system_prompt_template, user_prompt_template,
                                       args.model, model_config)

            saver.save_result(result, total_index - 1, total_stories)

            if 'error' in result:
                print(f"Error: {result['error']}")
                continue

            print(f"Story accuracy: {result['story_accuracy']:.2%}")
            for q_result in result['results']:
                print(f"  Q{q_result['question_num']}: {q_result['ground_truth']} → {q_result['prediction']} {'✓' if q_result['correct'] else '✗'}")
                total_questions += 1
                if q_result['correct']:
                    total_correct += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving final results...")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}. Saving results so far...")

    saver.finalize(total_questions, total_correct)

    if total_questions > 0:
        overall_accuracy = total_correct / total_questions
        print(f"\nOverall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})")
    else:
        print("\nNo questions were completed successfully.")


if __name__ == "__main__":
    main()
