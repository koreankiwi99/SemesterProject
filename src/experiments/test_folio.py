#!/usr/bin/env python3
"""
Group FOLIO questions by story_id to match LeanReasoner prompt format
(Updated: system → system prompt, input → user prompt, and num_stories=0 → all)
"""

import json
import argparse
from collections import defaultdict

def group_folio_by_story(folio_data):
    """Group FOLIO examples by story_id"""
    grouped = defaultdict(list)
    for example in folio_data:
        story_id = example.get('story_id')
        if story_id:
            grouped[story_id].append(example)
    return dict(grouped)

def build_prompts(story_examples):
    """Return (system_msg, user_prompt) for multiple questions on the same context"""
    premises = story_examples[0]["premises"]

    system_msg = (
        "You are a logician with a background in mathematics that translates natural language "
        "reasoning text to Lean code so that these natural language reasoning problems can be solved. "
        "During the translation, please pay close attention to the predicates and entities. "
        "There is an additional requirement: I also want you to try to prove the theorem you translated "
        "to Lean. If you can prove the theorem, give me True at the end of the answer. If you can prove "
        "the negation of the theorem, write False at the end of the answer. If you can neither prove the "
        "original theorem nor the negation of the theorem, please give me Unknown at the end of the answer."
    )

    lines = [f"Textual context: {premises}", ""]
    for i, example in enumerate(story_examples, 1):
        lines.append(
            f"Question {i}: Based on the above information, is the following statement true, false, or uncertain? "
            f"{example['conclusion']}"
        )
        lines.append("")
    lines.append("Please answer each question, and end each answer with one of: True, False, or Unknown.")
    user_prompt = "\n".join(lines)

    return system_msg, user_prompt

def load_and_group_folio(file_path):
    """Load FOLIO and group by story_id"""
    print(f"Loading FOLIO from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    grouped = group_folio_by_story(data)
    print(f"Found {len(grouped)} unique stories")
    print(f"Questions per story: min={min(len(v) for v in grouped.values())}, max={max(len(v) for v in grouped.values())}")
    return grouped

def test_gpt5_grouped(story_examples, api_key, model="gpt-5"):
    """Test GPT-5 on grouped questions"""
    import openai
    openai.api_key = api_key

    system_msg, user_prompt = build_prompts(story_examples)

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            #reasoning_effort="low",
        )

        gpt_response = response.choices[0].message.content.strip()
        answers = parse_multi_answers(gpt_response, len(story_examples))

        results = []
        for i, (example, prediction) in enumerate(zip(story_examples, answers)):
            correct = normalize_answer(prediction) == normalize_answer(example["label"])
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
            'gpt_response': gpt_response,
            'results': results,
            'story_accuracy': sum(r['correct'] for r in results) / len(results)
        }

    except Exception as e:
        return {'error': str(e)}

def parse_multi_answers(response, num_questions):
    """Extract multiple answers from GPT response"""
    answers = []
    lines = response.split('\n')
    for line in lines:
        low = line.strip().lower()
        if any(w in low for w in ['true', 'false', 'unknown', 'uncertain']):
            if 'true' in low and 'false' not in low:
                answers.append('True')
            elif 'false' in low and 'true' not in low:
                answers.append('False')
            elif 'unknown' in low or 'uncertain' in low:
                answers.append('Unknown')

    while len(answers) < num_questions:
        answers.append('Unknown')

    return answers[:num_questions]

def normalize_answer(answer):
    """Normalize answer format"""
    if not answer:
        return 'Unknown'
    low = answer.lower().strip()
    if low in ['true', 't']:
        return 'True'
    elif low in ['false', 'f']:
        return 'False'
    elif low in ['unknown', 'uncertain', 'u']:
        return 'Unknown'
    return answer

def save_all_responses(all_results, output_dir="results"):
    """Save all GPT-5 responses to files"""
    from datetime import datetime
    import os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    detailed_file = f"{output_dir}/gpt5_folio_responses_{timestamp}.json"
    with open(detailed_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    responses_dir = f"{output_dir}/gpt5_responses_{timestamp}"
    os.makedirs(responses_dir, exist_ok=True)

    for result in all_results:
        if 'error' not in result:
            story_id = result['story_id']
            response_file = f"{responses_dir}/story_{story_id}_response.txt"
            with open(response_file, 'w') as f:
                f.write(f"Story ID: {story_id}\n")
                f.write(f"Premises: {result['premises']}\n\n")
                f.write("=" * 50 + "\n")
                f.write("GPT-5 Full Response:\n")
                f.write("=" * 50 + "\n")
                f.write(result['gpt_response'])
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("Questions and Results:\n")
                f.write("=" * 50 + "\n")
                for q_result in result['results']:
                    f.write(f"Q{q_result['question_num']}: {q_result['conclusion']}\n")
                    f.write(f"Ground Truth: {q_result['ground_truth']}\n")
                    f.write(f"Prediction: {q_result['prediction']}\n")
                    f.write(f"Correct: {'Yes' if q_result['correct'] else 'No'}\n\n")

    print(f"Responses saved to: {detailed_file}")
    print(f"Individual responses saved to: {responses_dir}/")
    return detailed_file, responses_dir

def main():
    parser = argparse.ArgumentParser(description='Test GPT-5 on grouped FOLIO questions')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--folio_file', required=True, help='FOLIO JSON file')
    parser.add_argument('--num_stories', type=int, default=5,
                        help='Number of stories to test (0 or negative = all)')
    parser.add_argument('--output_dir', default='results', help='Directory to save responses')
    args = parser.parse_args()

    grouped_data = load_and_group_folio(args.folio_file)

    # NEW: allow all stories if num_stories <= 0
    if args.num_stories > 0:
        story_ids = list(grouped_data.keys())[:args.num_stories]
    else:
        story_ids = list(grouped_data.keys())

    print(f"\nTesting {len(story_ids)} stories with GPT-5")

    total_questions = 0
    total_correct = 0
    all_results = []

    for i, story_id in enumerate(story_ids):
        story_examples = grouped_data[story_id]
        print(f"\nStory {i+1} (ID: {story_id}): {len(story_examples)} questions")

        result = test_gpt5_grouped(story_examples, args.api_key)
        all_results.append(result)

        if 'error' in result:
            print(f"Error: {result['error']}")
            continue

        print(f"Story accuracy: {result['story_accuracy']:.2%}")
        for q_result in result['results']:
            print(f"  Q{q_result['question_num']}: {q_result['ground_truth']} → {q_result['prediction']} {'✓' if q_result['correct'] else '✗'}")
            total_questions += 1
            if q_result['correct']:
                total_correct += 1

    save_all_responses(all_results, args.output_dir)

    if total_questions > 0:
        overall_accuracy = total_correct / total_questions
        print(f"\nOverall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})")

if __name__ == "__main__":
    main()