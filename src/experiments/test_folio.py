#!/usr/bin/env python3
"""
Group FOLIO questions by story_id to match LeanReasoner prompt format
"""

import json
import argparse
import re
from collections import defaultdict
import os
from datetime import datetime

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
        "\n\nIMPORTANT: After your Lean analysis, provide your final answers in exactly this format:\n"
        "ANSWERS:\n"
        "1: True/False/Unknown\n"
        "2: True/False/Unknown\n"
        "3: True/False/Unknown\n"
        "(Use the exact number of questions provided)"
    )

    lines = [f"Textual context: {premises}", ""]
    for i, example in enumerate(story_examples, 1):
        lines.append(
            f"Question {i}: Based on the above information, is the following statement true, false, or uncertain? "
            f"{example['conclusion']}"
        )
        lines.append("")
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
        answers = parse_lean_response(gpt_response, len(story_examples))

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

def parse_lean_response(response, num_questions):
    """Extract multiple answers from GPT response with Lean format"""
    
    # Look for the structured ANSWERS: section first
    answers_match = re.search(r'ANSWERS:\s*\n(.*?)(?:\n\n|\n(?=[A-Za-z])|$)', response, re.DOTALL | re.IGNORECASE)
    if answers_match:
        answers_text = answers_match.group(1)
        # Extract numbered answers: "1: True" or "1: False" etc.
        numbered_answers = re.findall(r'(\d+):\s*(True|False|Unknown)', answers_text, re.IGNORECASE)
        if len(numbered_answers) >= num_questions:
            # Sort by question number and extract answers
            numbered_answers.sort(key=lambda x: int(x[0]))
            return [normalize_answer(match[1]) for match in numbered_answers[:num_questions]]
    
    # Fallback 1: Look for consecutive True/False/Unknown lines
    consecutive_answers = []
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if re.match(r'^(True|False|Unknown)$', line, re.IGNORECASE):
            consecutive_answers.append(line)
            if len(consecutive_answers) == num_questions:
                return [normalize_answer(a) for a in consecutive_answers]
        else:
            # Reset if we hit a non-answer line (unless we already have some answers)
            if consecutive_answers and len(consecutive_answers) < num_questions:
                consecutive_answers = []
    
    # Fallback 2: Extract any True/False/Unknown in order
    all_answers = re.findall(r'\b(True|False|Unknown)\b', response, re.IGNORECASE)
    if len(all_answers) >= num_questions:
        return [normalize_answer(a) for a in all_answers[:num_questions]]
    
    # Pad with Unknown if we don't have enough answers
    result = [normalize_answer(a) for a in all_answers]
    while len(result) < num_questions:
        result.append('Unknown')
    
    return result[:num_questions]

def parse_multi_answers(response, num_questions):
    """Legacy parsing function - kept for backward compatibility"""
    return parse_lean_response(response, num_questions)

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

class IncrementalSaver:
    """Handles incremental saving of results"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output files
        self.detailed_file = f"{output_dir}/gpt5_folio_responses_{self.timestamp}.json"
        self.responses_dir = f"{output_dir}/gpt5_responses_{self.timestamp}"
        self.progress_file = f"{output_dir}/progress_{self.timestamp}.txt"
        
        os.makedirs(self.responses_dir, exist_ok=True)
        
        # Initialize files
        self._init_files()
    
    def _init_files(self):
        """Initialize output files"""
        # Initialize JSON file with empty array
        with open(self.detailed_file, 'w') as f:
            json.dump([], f)
        
        # Initialize progress file
        with open(self.progress_file, 'w') as f:
            f.write(f"FOLIO Testing Progress - Started at {self.timestamp}\n")
            f.write("=" * 50 + "\n")
    
    def save_result(self, result, story_index, total_stories):
        """Save a single result incrementally"""
        # 1. Append to detailed JSON file
        self._append_to_json(result)
        
        # 2. Save individual response file
        if 'error' not in result:
            self._save_individual_response(result)
        
        # 3. Update progress file
        self._update_progress(result, story_index, total_stories)
        
        print(f"✓ Saved result for story {result.get('story_id', 'unknown')}")
    
    def _append_to_json(self, result):
        """Append result to JSON file"""
        try:
            # Read existing data
            with open(self.detailed_file, 'r') as f:
                data = json.load(f)
            
            # Append new result
            data.append(result)
            
            # Write back
            with open(self.detailed_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update JSON file: {e}")
    
    def _save_individual_response(self, result):
        """Save individual response file"""
        story_id = result['story_id']
        response_file = f"{self.responses_dir}/story_{story_id}_response.txt"
        
        try:
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
        except Exception as e:
            print(f"Warning: Could not save individual response: {e}")
    
    def _update_progress(self, result, story_index, total_stories):
        """Update progress file"""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"\nStory {story_index+1}/{total_stories}: {result.get('story_id', 'unknown')}\n")
                if 'error' in result:
                    f.write(f"ERROR: {result['error']}\n")
                else:
                    accuracy = result['story_accuracy']
                    num_questions = len(result['results'])
                    correct_count = sum(r['correct'] for r in result['results'])
                    f.write(f"Accuracy: {correct_count}/{num_questions} ({accuracy:.2%})\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")
    
    def finalize(self, total_questions, total_correct):
        """Write final summary"""
        try:
            with open(self.progress_file, 'a') as f:
                f.write("\n" + "=" * 50 + "\n")
                f.write("FINAL RESULTS\n")
                f.write("=" * 50 + "\n")
                if total_questions > 0:
                    overall_accuracy = total_correct / total_questions
                    f.write(f"Overall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})\n")
                else:
                    f.write("No questions completed successfully.\n")
                f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"Warning: Could not finalize progress: {e}")
        
        print(f"\nAll results saved to: {self.detailed_file}")
        print(f"Individual responses in: {self.responses_dir}/")
        print(f"Progress log: {self.progress_file}")

def main():
    parser = argparse.ArgumentParser(description='Test GPT-5 on grouped FOLIO questions with incremental saving')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--folio_file', required=True, help='FOLIO JSON file')
    parser.add_argument('--num_stories', type=int, default=5,
                        help='Number of stories to test (0 or negative = all)')
    parser.add_argument('--output_dir', default='results', help='Directory to save responses')
    args = parser.parse_args()

    grouped_data = load_and_group_folio(args.folio_file)

    # Select stories to test
    if args.num_stories > 0:
        story_ids = list(grouped_data.keys())[:args.num_stories]
    else:
        story_ids = list(grouped_data.keys())

    print(f"\nTesting {len(story_ids)} stories with GPT-5")  
    # Initialize incremental saver
    saver = IncrementalSaver(args.output_dir)
    
    total_questions = 0
    total_correct = 0

    try:
        for i, story_id in enumerate(story_ids):
            story_examples = grouped_data[story_id]
            print(f"\nStory {i+1}/{len(story_ids)} (ID: {story_id}): {len(story_examples)} questions")

            result = test_gpt5_grouped(story_examples, args.api_key)
            
            # Save immediately after each result
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

    # Finalize results
    saver.finalize(total_questions, total_correct)

    if total_questions > 0:
        overall_accuracy = total_correct / total_questions
        print(f"\nOverall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})")
    else:
        print("\nNo questions were completed successfully.")

if __name__ == "__main__":
    main()