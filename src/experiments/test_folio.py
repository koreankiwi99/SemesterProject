#!/usr/bin/env python3
"""
Group FOLIO questions by story_id and test with customizable prompts.
Tests multiple questions per story in a single LLM call.
"""

import argparse

from utils.prompts import load_prompt
from utils.answer_parsing import parse_folio_multiple_answers, normalize_answer
from utils.savers import FOLIOSaver
from datasets.folio import load_and_group_folio, build_folio_prompt_grouped


def test_model_grouped(story_examples, api_key, system_prompt_template, user_prompt_template, model="gpt-4"):
    """Test model on grouped questions from a single story.

    Args:
        story_examples: List of FOLIO examples from the same story
        api_key: OpenAI API key
        system_prompt_template: System prompt template
        user_prompt_template: User prompt template
        model: Model name to use

    Returns:
        dict: Results for this story including all questions
    """
    import openai
    openai.api_key = api_key

    system_msg, user_prompt = build_folio_prompt_grouped(story_examples, system_prompt_template, user_prompt_template)

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
        )

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
    parser.add_argument('--model', default='gpt-4', help='Model to use')

    args = parser.parse_args()

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

    print(f"\nTesting {len(story_ids)} stories with {args.model}")

    saver = FOLIOSaver(args.output_dir, args.prompt_name)

    total_questions = 0
    total_correct = 0

    try:
        for i, story_id in enumerate(story_ids):
            story_examples = grouped_data[story_id]
            print(f"\nStory {i+1}/{len(story_ids)} (ID: {story_id}): {len(story_examples)} questions")

            result = test_model_grouped(story_examples, args.api_key,
                                       system_prompt_template, user_prompt_template,
                                       args.model)

            saver.save_result(result, i, len(story_ids))

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
