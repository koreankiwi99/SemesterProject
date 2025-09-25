#!/usr/bin/env python3
"""
Test GPT-5 on ProofWriter dataset with Lean reasoning prompts
"""

import json
import argparse
import re
from collections import defaultdict
import os
from datetime import datetime

def load_proofwriter_dataset(data_file):
    """Load ProofWriter dataset from JSONL file"""
    print(f"Loading ProofWriter from: {data_file}")
    
    examples = []
    with open(data_file, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples")
    return examples

def build_proofwriter_prompt(example):
    """Build prompt for a single ProofWriter example with Lean reasoning format"""
    # Extract theory (facts and rules)
    theory = example.get("theory", "")
    
    system_msg = (
        "You are a logician with a background in mathematics that translates natural language "
        "reasoning text to Lean code so that these natural language reasoning problems can be solved. "
        "During the translation, please pay close attention to the predicates and entities. "
        "There is an additional requirement: I also want you to try to prove the theorem you translated "
        "to Lean. If you can prove the theorem, give me True at the end of the answer. If you can prove "
        "the negation of the theorem, write False at the end of the answer. If you can neither prove the "
        "original theorem nor the negation of the theorem, please give me Unknown at the end of the answer."
    )
    
    # Select a random question from the example
    questions = example.get("questions", {})
    if not questions:
        raise ValueError("No questions found in example")
    
    # Get first question for single-question testing
    question_key = list(questions.keys())[0]
    question_data = questions[question_key]
    question_text = question_data["question"]
    
    user_prompt = f"""Input:
Textual context: {theory}
Question: Based on the above information, is the following statement true, false, or unknown? {question_text}"""
    
    return system_msg, user_prompt, question_data

def test_gpt5_proofwriter(example, api_key, model="gpt-5"):
    """Test GPT-5 on a single ProofWriter example"""
    import openai
    openai.api_key = api_key

    try:
        system_msg, user_prompt, question_data = build_proofwriter_prompt(example)
    except ValueError as e:
        return {'error': str(e), 'example_id': example.get('id', 'unknown')}

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for more consistent reasoning
        )

        gpt_response = response.choices[0].message.content.strip()
        prediction = parse_single_answer(gpt_response)

        # ProofWriter uses boolean or "Unknown" for answers
        ground_truth = question_data.get('answer')
        if ground_truth is True:
            ground_truth = 'True'
        elif ground_truth is False:
            ground_truth = 'False'
        else:
            ground_truth = 'Unknown'
        
        correct = normalize_proofwriter_answer(prediction) == normalize_proofwriter_answer(ground_truth)
        
        return {
            'example_id': example.get('id', 'unknown'),
            'theory': example.get('theory', ''),
            'question': question_data["question"],
            'ground_truth': ground_truth,
            'prediction': prediction,
            'correct': correct,
            'gpt_response': gpt_response,
            'qdep': question_data.get('QDep', 0),
            'strategy': question_data.get('strategy', 'unknown')
        }

    except Exception as e:
        return {'error': str(e), 'example_id': example.get('id', 'unknown')}

def parse_single_answer(response):
    """Extract single answer from GPT response"""
    
    # Look for final answer at the end
    lines = response.strip().split('\n')
    
    # Check the last few lines for True/False/Unknown
    for line in reversed(lines[-5:]):  # Check last 5 lines
        line = line.strip()
        if re.match(r'^(True|False|Unknown)$', line, re.IGNORECASE):
            return normalize_proofwriter_answer(line)
    
    # Look for any True/False/Unknown in the response
    all_answers = re.findall(r'\b(True|False|Unknown)\b', response, re.IGNORECASE)
    if all_answers:
        return normalize_proofwriter_answer(all_answers[-1])  # Take the last one
    
    return 'Unknown'  # Default fallback

def normalize_proofwriter_answer(answer):
    """Normalize ProofWriter answer format"""
    if not answer:
        return 'Unknown'
    
    if isinstance(answer, bool):
        return 'True' if answer else 'False'
    
    answer_str = str(answer).lower().strip()
    
    if answer_str in ['true', 't', '1', 'yes']:
        return 'True'
    elif answer_str in ['false', 'f', '0', 'no']:
        return 'False'
    else:
        return 'Unknown'

class IncrementalSaver:
    """Handles incremental saving of results"""
    
    def __init__(self, output_dir="proofwriter_results"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output files
        self.detailed_file = f"{output_dir}/proofwriter_gpt5_responses_{self.timestamp}.json"
        self.responses_dir = f"{output_dir}/proofwriter_gpt5_responses_{self.timestamp}"
        self.progress_file = f"{output_dir}/proofwriter_progress_{self.timestamp}.txt"
        
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
            f.write(f"ProofWriter Testing Progress - Started at {self.timestamp}\n")
            f.write("=" * 50 + "\n")
    
    def save_result(self, result, example_index, total_examples):
        """Save a single result incrementally"""
        # 1. Append to detailed JSON file
        self._append_to_json(result)
        
        # 2. Save individual response file
        if 'error' not in result:
            self._save_individual_response(result)
        
        # 3. Update progress file
        self._update_progress(result, example_index, total_examples)
        
        print(f"✓ Saved result for example {result.get('example_id', 'unknown')}")
    
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
        example_id = result['example_id']
        response_file = f"{self.responses_dir}/example_{example_id}_response.txt"
        
        try:
            with open(response_file, 'w') as f:
                f.write(f"Example ID: {example_id}\n")
                f.write(f"Theory: {result['theory']}\n\n")
                f.write(f"Question: {result['question']}\n")
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Prediction: {result['prediction']}\n")
                f.write(f"Correct: {'Yes' if result['correct'] else 'No'}\n")
                f.write(f"QDep: {result.get('qdep', 'N/A')}\n")
                f.write(f"Strategy: {result.get('strategy', 'N/A')}\n\n")
                f.write("=" * 50 + "\n")
                f.write("GPT-5 Full Response:\n")
                f.write("=" * 50 + "\n")
                f.write(result['gpt_response'])
        except Exception as e:
            print(f"Warning: Could not save individual response: {e}")
    
    def _update_progress(self, result, example_index, total_examples):
        """Update progress file"""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"\nExample {example_index+1}/{total_examples}: {result.get('example_id', 'unknown')}\n")
                if 'error' in result:
                    f.write(f"ERROR: {result['error']}\n")
                else:
                    f.write(f"Correct: {'Yes' if result['correct'] else 'No'}\n")
                    f.write(f"Ground Truth: {result['ground_truth']} -> Prediction: {result['prediction']}\n")
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
    parser = argparse.ArgumentParser(description='Test GPT-5 on ProofWriter dataset with Lean reasoning')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--data_file', required=True, help='ProofWriter JSONL file (e.g., meta-train.jsonl)')
    parser.add_argument('--num_examples', type=int, default=100,
                        help='Number of examples to test (0 or negative = all)')
    parser.add_argument('--output_dir', default='proofwriter_results', help='Directory to save responses')
    parser.add_argument('--model', default='gpt-5', help='OpenAI model to use')
    args = parser.parse_args()

    # Load ProofWriter data
    examples = load_proofwriter_dataset(args.data_file)

    # Select examples to test
    if args.num_examples > 0:
        examples = examples[:args.num_examples]

    print(f"\nTesting {len(examples)} examples with {args.model}")  
    
    # Initialize incremental saver
    saver = IncrementalSaver(args.output_dir)
    
    total_questions = 0
    total_correct = 0

    try:
        for i, example in enumerate(examples):
            print(f"\nExample {i+1}/{len(examples)} (ID: {example.get('id', 'unknown')})")

            result = test_gpt5_proofwriter(example, args.api_key, args.model)
            
            # Save immediately after each result
            saver.save_result(result, i, len(examples))

            if 'error' in result:
                print(f"Error: {result['error']}")
                continue

            print(f"  {result['ground_truth']} → {result['prediction']} {'✓' if result['correct'] else '✗'}")
            total_questions += 1
            if result['correct']:
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