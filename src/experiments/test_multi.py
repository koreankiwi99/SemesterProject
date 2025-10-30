#!/usr/bin/env python3
"""
Test models on Multi-LogiEval dataset using the paper's evaluation approach.
Samples individual questions rather than files for balanced evaluation.
"""

import argparse
import json
import os
from collections import defaultdict

from utils.prompts import load_prompt
from utils.answer_parsing import parse_multilogieval_answer, normalize_answer
from utils.savers import MultiLogiEvalSaver
from datasets.multilogieval import load_and_sample_multilogieval, build_multilogieval_prompt


def test_model_on_samples(samples, api_key, system_prompt_template, user_prompt_template,
                          model="gpt-5", model_config=None, saver=None):
    """Test model on sampled questions.

    Args:
        samples: List of question samples
        api_key: OpenAI API key
        system_prompt_template: System prompt template
        user_prompt_template: User prompt template
        model: Model name to use
        model_config: Optional model configuration dict (temperature, reasoning_effort, etc.)
        saver: MultiLogiEvalSaver instance for incremental saving

    Returns:
        tuple: (results, results_by_combination)
    """
    import openai
    openai.api_key = api_key

    results = []
    results_by_combination = defaultdict(list)

    print(f"\nTesting {len(samples)} questions with {model}...")
    print("=" * 70)

    for i, sample in enumerate(samples, 1):
        combination_key = (sample['logic_type'], sample['depth_dir'])
        print(f"\n[{i}/{len(samples)}] {sample['logic_type']}/{sample['depth_dir']} - {sample['rule']}")

        system_msg, user_prompt = build_multilogieval_prompt(sample, system_prompt_template, user_prompt_template)

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

            gpt_response = response.choices[0].message.content.strip()

            # Parse the answer
            prediction = parse_multilogieval_answer(gpt_response)
            ground_truth = normalize_answer(sample['answer'], answer_format="yes_no")

            correct = prediction == ground_truth

            result = {
                'question_num': i,
                'logic_type': sample['logic_type'],
                'depth': sample['depth'],
                'depth_dir': sample['depth_dir'],
                'rule': sample['rule'],
                'context': sample['context'],
                'question': sample['question'],
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': correct,
                'system_prompt': system_msg,
                'user_prompt': user_prompt,
                'model': model,
                'model_config': model_config or {},
                'full_response': gpt_response,
                'source_file': sample['source_file']
            }

            results.append(result)
            results_by_combination[combination_key].append(result)

            # Save incrementally if saver provided
            if saver:
                saver.save_result(result, i - 1, len(samples))

            print(f"  Prediction: {prediction} | Ground Truth: {ground_truth} | {'✓' if correct else '✗'}")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            result = {
                'question_num': i,
                'logic_type': sample['logic_type'],
                'depth': sample['depth'],
                'depth_dir': sample['depth_dir'],
                'rule': sample['rule'],
                'context': sample['context'],
                'question': sample['question'],
                'ground_truth': normalize_answer(sample['answer'], answer_format="yes_no"),
                'prediction': 'Error',
                'correct': False,
                'error': str(e),
                'source_file': sample['source_file']
            }
            results.append(result)
            results_by_combination[combination_key].append(result)

            # Save error incrementally if saver provided
            if saver:
                saver.save_result(result, i - 1, len(samples))

    return results, results_by_combination


def main():
    parser = argparse.ArgumentParser(
        description='Test models on Multi-LogiEval dataset (sampling individual questions)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Zero-shot CoT (paper's main approach) - 10 questions per combination
  python test_multi.py \\
      --api_key YOUR_KEY \\
      --data_dir data/multi_logi_original/data \\
      --system_prompt prompts/multilogi/zero_shot_cot_system.txt \\
      --user_prompt prompts/multilogi/zero_shot_cot_user.txt \\
      --prompt_name zero_shot_cot \\
      --samples_per_combination 10 \\
      --model gpt-4
        """
    )
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--data_dir', required=True, help='Path to Multi-LogiEval data directory')
    parser.add_argument('--system_prompt', required=True, help='Path to system prompt file')
    parser.add_argument('--user_prompt', required=True, help='Path to user prompt template file')
    parser.add_argument('--prompt_name', default='test', help='Name for output files')
    parser.add_argument('--logic_types', nargs='+',
                        default=['fol', 'nm', 'pl'],
                        help='Logic types to test (default: all)')
    parser.add_argument('--depths', nargs='+',
                        default=['d1_Data', 'd2_Data', 'd3_Data', 'd4_Data', 'd5_Data'],
                        help='Depths to test (default: all)')
    parser.add_argument('--samples_per_combination', type=int, default=10,
                        help='Number of questions to sample per logic type × depth combination (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
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
    print("Loading prompts...")
    system_prompt_template = load_prompt(args.system_prompt)
    user_prompt_template = load_prompt(args.user_prompt)
    print(f"✓ Loaded system prompt from {args.system_prompt}")
    print(f"✓ Loaded user prompt from {args.user_prompt}")

    # Load and sample dataset
    sampled_questions = load_and_sample_multilogieval(
        args.data_dir,
        args.logic_types,
        args.depths,
        args.samples_per_combination,
        args.seed
    )

    if not sampled_questions:
        print("No questions found! Check your data directory and filters.")
        return

    # Handle resume functionality
    processed_keys = set()
    if args.resume:
        all_results_file = os.path.join(args.resume, 'all_results.json')
        if os.path.exists(all_results_file):
            print(f"\nResuming from: {args.resume}")
            with open(all_results_file, 'r') as f:
                previous_results = json.load(f)
            # Track by (logic_type, depth_dir, rule, context, question) to identify unique questions
            processed_keys = {
                (r['logic_type'], r['depth_dir'], r['rule'], r['context'], r['question'])
                for r in previous_results if 'logic_type' in r
            }
            print(f"Found {len(processed_keys)} already processed questions")
        else:
            print(f"Warning: Resume directory exists but no all_results.json found: {all_results_file}")

    # Filter out already processed questions
    remaining_questions = [
        q for q in sampled_questions
        if (q['logic_type'], q['depth_dir'], q['rule'], q['context'], q['question']) not in processed_keys
    ]

    print(f"\nTotal questions to test: {len(sampled_questions)}")
    if processed_keys:
        print(f"Already completed: {len(processed_keys)}")
        print(f"Remaining: {len(remaining_questions)}")

    if not remaining_questions:
        print("\nAll questions already completed!")
        return

    # Initialize saver (will resume if --resume provided)
    if args.resume:
        saver = MultiLogiEvalSaver(args.output_dir, args.prompt_name, resume_dir=args.resume)
    else:
        saver = MultiLogiEvalSaver(args.output_dir, args.prompt_name)

    # Test model on remaining questions
    try:
        results, results_by_combination = test_model_on_samples(
            remaining_questions,
            args.api_key,
            system_prompt_template,
            user_prompt_template,
            args.model,
            model_config,
            saver  # Pass saver for incremental saving
        )

        # Finalize and save summaries
        # Load all results (including previous ones if resuming)
        with open(saver.all_results_file, 'r') as f:
            all_results = json.load(f)

        # Rebuild results_by_combination from all results
        all_results_by_combination = defaultdict(list)
        for r in all_results:
            key = (r['logic_type'], r['depth_dir'])
            all_results_by_combination[key].append(r)

        saver.finalize(all_results, all_results_by_combination)

        # Print final summary
        total_correct = sum(r['correct'] for r in all_results)
        total_questions = len(all_results)
        if total_questions > 0:
            overall_accuracy = total_correct / total_questions
            print(f"\n{'='*70}")
            print(f"OVERALL ACCURACY: {total_correct}/{total_questions} ({overall_accuracy:.2%})")
            print(f"{'='*70}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
