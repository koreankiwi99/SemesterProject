#!/usr/bin/env python3
"""
Test GPT-5 on Multi-LogiEval dataset with incremental saving and organized results
"""

import json
import argparse
import re
from collections import defaultdict
import os
from datetime import datetime
from pathlib import Path

def load_multilogieval_file(file_path):
    """Load a single Multi-LogiEval JSON file"""
    try:
        # Try UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        # If UTF-8 fails, try with latin-1 or other encodings
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                data = json.load(f)
        except:
            # As a last resort, try with errors='ignore'
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
    
    # Extract metadata
    logic_type = data.get('logic', 'unknown')
    rule = data.get('rule', 'unknown')
    depth = data.get('depth', 'unknown')
    samples = data.get('samples', [])
    
    # Convert samples to a common format
    converted_samples = []
    for sample in samples:
        converted_samples.append({
            'id': sample.get('id'),
            'context': sample.get('context', ''),
            'question': sample.get('question', ''),
            'answer': sample.get('answer', ''),
            'logic_type': logic_type,
            'rule': rule,
            'depth': depth,
            'file_path': str(file_path)
        })
    
    return {
        'logic_type': logic_type,
        'rule': rule,
        'depth': depth,
        'samples': converted_samples,
        'file_path': str(file_path)
    }

def load_multilogieval_dataset(data_dir, logic_types=None, depths=None):
    """
    Load Multi-LogiEval dataset from directory structure
    
    Args:
        data_dir: Path to data directory
        logic_types: List of logic types to load (e.g., ['fol', 'pl', 'nm'])
        depths: List of depths to load (e.g., ['d1_Data', 'd2_Data'])
    """
    data_path = Path(data_dir)
    all_files = []
    
    # Default to all logic types if not specified
    # The actual directory names are 'fol', 'nm', 'pl'
    if logic_types is None:
        logic_types = ['fol', 'nm', 'pl']
    
    # Default to all depths if not specified
    if depths is None:
        depths = ['d1_Data', 'd2_Data', 'd3_Data', 'd4_Data', 'd5_Data']
    
    print(f"Loading Multi-LogiEval from: {data_dir}")
    print(f"Logic types: {logic_types}")
    print(f"Depths: {depths}")
    
    # Iterate through depth directories
    for depth_dir in depths:
        depth_path = data_path / depth_dir
        if not depth_path.exists():
            print(f"Warning: {depth_path} does not exist, skipping...")
            continue
        
        # Iterate through logic type directories
        for logic_type in logic_types:
            logic_path = depth_path / logic_type
            if not logic_path.exists():
                print(f"Warning: {logic_path} does not exist, skipping...")
                continue
            
            # Load all JSON files in this directory
            json_files = list(logic_path.glob('*.json'))
            print(f"Found {len(json_files)} files in {depth_dir}/{logic_type}")
            
            for json_file in json_files:
                file_data = load_multilogieval_file(json_file)
                all_files.append(file_data)
    
    print(f"\nTotal files loaded: {len(all_files)}")
    total_samples = sum(len(f['samples']) for f in all_files)
    print(f"Total samples: {total_samples}")
    
    return all_files

def build_prompt_single_sample(sample):
    """Build prompt for a single Multi-LogiEval sample"""
    
    system_msg = (
        "You are a logician with a background in mathematics that translates natural language "
        "reasoning text to Lean code so that these natural language reasoning problems can be solved. "
        "During the translation, please pay close attention to the predicates and entities. "
        "There is an additional requirement: I also want you to try to prove the theorem you translated "
        "to Lean. If you can prove the theorem, give me True at the end of the answer. If you can prove "
        "the negation of the theorem, write False at the end of the answer. If you can neither prove the "
        "original theorem nor the negation of the theorem, please give me Unknown at the end of the answer."
        "\n\nIMPORTANT: After your Lean analysis, provide your final answer in exactly this format:\n"
        "ANSWER: True/False/Unknown"
    )
    
    user_prompt = f"Textual context: {sample['context']}\n\nQuestion: {sample['question']}"
    
    return system_msg, user_prompt

def test_gpt5_multilogieval(file_data, api_key, model="gpt-5"):
    """Test GPT-5 on Multi-LogiEval file - processing each sample individually"""
    import openai
    openai.api_key = api_key
    
    samples = file_data['samples']
    
    if not samples:
        return {'error': 'No samples in file'}
    
    results = []
    all_responses = []
    
    # Process each sample individually since they have different contexts
    for i, sample in enumerate(samples, 1):
        print(f"  Processing sample {i}/{len(samples)}...", end=' ')
        
        system_msg, user_prompt = build_prompt_single_sample(sample)
        
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt}
                ],
            )
            
            gpt_response = response.choices[0].message.content.strip()
            all_responses.append({
                'sample_id': sample.get('id'),
                'response': gpt_response
            })
            
            # Parse single answer
            prediction = parse_single_answer(gpt_response)
            
            # Normalize answer format (yes/no to True/False if needed)
            ground_truth = normalize_answer(sample['answer'])
            prediction_normalized = normalize_answer(prediction)
            
            correct = prediction_normalized == ground_truth
            
            results.append({
                'question_num': i,
                'sample_id': sample.get('id'),
                'context': sample['context'],
                'question': sample['question'],
                'ground_truth': ground_truth,
                'prediction': prediction_normalized,
                'correct': correct
            })
            
            print(f"{'✓' if correct else '✗'}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results.append({
                'question_num': i,
                'sample_id': sample.get('id'),
                'context': sample['context'],
                'question': sample['question'],
                'ground_truth': normalize_answer(sample['answer']),
                'prediction': 'Error',
                'correct': False,
                'error': str(e)
            })
    
    return {
        'logic_type': file_data['logic_type'],
        'rule': file_data['rule'],
        'depth': file_data['depth'],
        'file_path': file_data['file_path'],
        'all_responses': all_responses,
        'results': results,
        'file_accuracy': sum(r['correct'] for r in results) / len(results) if results else 0
    }

def parse_single_answer(response):
    """Extract single answer from GPT response"""
    
    # Look for ANSWER: format first
    answer_match = re.search(r'ANSWER:\s*(True|False|Unknown|Yes|No)', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1)
    
    # Look for final True/False/Unknown/Yes/No
    final_answer = re.findall(r'\b(True|False|Unknown|Yes|No)\b', response, re.IGNORECASE)
    if final_answer:
        return final_answer[-1]  # Take the last occurrence
    
    return 'Unknown'

def parse_lean_response(response, num_questions):
    """Extract multiple answers from GPT response with Lean format"""
    
    # Look for the structured ANSWERS: section first
    answers_match = re.search(r'ANSWERS:\s*\n(.*?)(?:\n\n|\n(?=[A-Za-z])|$)', response, re.DOTALL | re.IGNORECASE)
    if answers_match:
        answers_text = answers_match.group(1)
        numbered_answers = re.findall(r'(\d+):\s*(True|False|Unknown)', answers_text, re.IGNORECASE)
        if len(numbered_answers) >= num_questions:
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

def normalize_answer(answer):
    """Normalize answer format - handles True/False/Unknown and Yes/No"""
    if not answer:
        return 'Unknown'
    low = answer.lower().strip()
    if low in ['true', 't', 'yes', 'y']:
        return 'True'
    elif low in ['false', 'f', 'no', 'n']:
        return 'False'
    elif low in ['unknown', 'uncertain', 'u']:
        return 'Unknown'
    return answer

class IncrementalSaver:
    """Handles incremental saving of results with organized structure"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create main output files
        self.detailed_file = f"{output_dir}/gpt5_multilogieval_all_results_{self.timestamp}.json"
        self.progress_file = f"{output_dir}/progress_{self.timestamp}.txt"
        
        # Create organized directory structure
        self.responses_dir = f"{output_dir}/responses_{self.timestamp}"
        self.by_logic_dir = f"{output_dir}/by_logic_{self.timestamp}"
        self.by_depth_dir = f"{output_dir}/by_depth_{self.timestamp}"
        self.summary_dir = f"{output_dir}/summaries_{self.timestamp}"
        
        os.makedirs(self.responses_dir, exist_ok=True)
        os.makedirs(self.by_logic_dir, exist_ok=True)
        os.makedirs(self.by_depth_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        
        # Track results for aggregation
        self.results_by_logic = defaultdict(list)
        self.results_by_depth = defaultdict(list)
        self.results_by_logic_depth = defaultdict(lambda: defaultdict(list))
        
        # Initialize files
        self._init_files()
    
    def _init_files(self):
        """Initialize output files"""
        with open(self.detailed_file, 'w') as f:
            json.dump([], f)
        
        with open(self.progress_file, 'w') as f:
            f.write(f"Multi-LogiEval Testing Progress - Started at {self.timestamp}\n")
            f.write("=" * 70 + "\n\n")
    
    def save_result(self, result, file_index, total_files):
        """Save a single result incrementally"""
        # 1. Append to main detailed JSON file
        self._append_to_json(result)
        
        # 2. Save individual response file
        if 'error' not in result:
            self._save_individual_response(result)
            
            # 3. Track for aggregated results
            self._track_for_aggregation(result)
        
        # 4. Update progress file
        self._update_progress(result, file_index, total_files)
        
        print(f"✓ Saved result for {result.get('logic_type', 'unknown')}/{result.get('rule', 'unknown')}")
    
    def _append_to_json(self, result):
        """Append result to main JSON file"""
        try:
            with open(self.detailed_file, 'r') as f:
                data = json.load(f)
            
            data.append(result)
            
            with open(self.detailed_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update JSON file: {e}")
    
    def _save_individual_response(self, result):
        """Save individual response file"""
        logic_type = result['logic_type']
        rule = result['rule']
        depth = result['depth']
        
        # Create safe filename
        safe_rule = re.sub(r'[^\w\-_]', '_', rule)
        response_file = f"{self.responses_dir}/{logic_type}_{depth}_{safe_rule}.txt"
        
        try:
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write(f"Logic Type: {logic_type}\n")
                f.write(f"Rule: {rule}\n")
                f.write(f"Depth: {depth}\n")
                f.write(f"File: {result['file_path']}\n")
                f.write(f"Accuracy: {result['file_accuracy']:.2%}\n")
                f.write("=" * 70 + "\n\n")
                
                # Each sample was processed individually
                f.write("=" * 70 + "\n")
                f.write("Individual Sample Results:\n")
                f.write("=" * 70 + "\n\n")
                
                for q_result in result['results']:
                    f.write(f"Sample {q_result['question_num']} (ID: {q_result.get('sample_id', 'N/A')}):\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Context: {q_result.get('context', 'N/A')}\n\n")
                    f.write(f"Question: {q_result['question']}\n\n")
                    
                    # Find the corresponding response
                    if 'all_responses' in result:
                        matching_response = next(
                            (r for r in result['all_responses'] if r['sample_id'] == q_result.get('sample_id')),
                            None
                        )
                        if matching_response:
                            f.write(f"GPT Response:\n{matching_response['response']}\n\n")
                    
                    f.write(f"Ground Truth: {q_result['ground_truth']}\n")
                    f.write(f"Prediction:   {q_result['prediction']}\n")
                    f.write(f"Correct:      {'✓ Yes' if q_result['correct'] else '✗ No'}\n")
                    
                    if 'error' in q_result:
                        f.write(f"Error: {q_result['error']}\n")
                    
                    f.write("\n" + "=" * 70 + "\n\n")
        except Exception as e:
            print(f"Warning: Could not save individual response: {e}")
    
    def _track_for_aggregation(self, result):
        """Track results for later aggregation"""
        logic_type = result['logic_type']
        depth = result['depth']
        
        self.results_by_logic[logic_type].append(result)
        self.results_by_depth[depth].append(result)
        self.results_by_logic_depth[logic_type][depth].append(result)
    
    def _update_progress(self, result, file_index, total_files):
        """Update progress file"""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"[{file_index+1}/{total_files}] {result.get('logic_type')}/{result.get('depth')}/{result.get('rule')}\n")
                
                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    accuracy = result['file_accuracy']
                    num_questions = len(result['results'])
                    correct_count = sum(r['correct'] for r in result['results'])
                    f.write(f"  Accuracy: {correct_count}/{num_questions} ({accuracy:.2%})\n")
                
                f.write(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")
    
    def _save_aggregated_results(self):
        """Save aggregated results by logic type and depth"""
        print("\nGenerating aggregated summaries...")
        
        # 1. Summary by logic type
        self._save_logic_summaries()
        
        # 2. Summary by depth
        self._save_depth_summaries()
        
        # 3. Summary by logic type AND depth
        self._save_logic_depth_summaries()
        
        # 4. Overall summary
        self._save_overall_summary()
    
    def _save_logic_summaries(self):
        """Save summaries organized by logic type"""
        for logic_type, results in self.results_by_logic.items():
            safe_logic = re.sub(r'[^\w\-_]', '_', logic_type)
            
            # JSON file
            json_file = f"{self.by_logic_dir}/{safe_logic}_results.json"
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Summary text file
            summary_file = f"{self.by_logic_dir}/{safe_logic}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Summary for Logic Type: {logic_type}\n")
                f.write("=" * 70 + "\n\n")
                
                total_questions = sum(len(r['results']) for r in results)
                total_correct = sum(sum(q['correct'] for q in r['results']) for r in results)
                
                f.write(f"Total Files: {len(results)}\n")
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Overall Accuracy: {total_correct/total_questions:.2%}\n\n")
                
                # By depth within this logic type
                f.write("Breakdown by Depth:\n")
                f.write("-" * 70 + "\n")
                depth_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
                for r in results:
                    depth = r['depth']
                    for q in r['results']:
                        depth_stats[depth]['total'] += 1
                        if q['correct']:
                            depth_stats[depth]['correct'] += 1
                
                for depth in sorted(depth_stats.keys()):
                    stats = depth_stats[depth]
                    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                    f.write(f"  {depth}: {stats['correct']}/{stats['total']} ({acc:.2%})\n")
    
    def _save_depth_summaries(self):
        """Save summaries organized by depth"""
        for depth, results in self.results_by_depth.items():
            safe_depth = re.sub(r'[^\w\-_]', '_', depth)
            
            # JSON file
            json_file = f"{self.by_depth_dir}/{safe_depth}_results.json"
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Summary text file
            summary_file = f"{self.by_depth_dir}/{safe_depth}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Summary for Depth: {depth}\n")
                f.write("=" * 70 + "\n\n")
                
                total_questions = sum(len(r['results']) for r in results)
                total_correct = sum(sum(q['correct'] for q in r['results']) for r in results)
                
                f.write(f"Total Files: {len(results)}\n")
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Overall Accuracy: {total_correct/total_questions:.2%}\n\n")
                
                # By logic type within this depth
                f.write("Breakdown by Logic Type:\n")
                f.write("-" * 70 + "\n")
                logic_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
                for r in results:
                    logic = r['logic_type']
                    for q in r['results']:
                        logic_stats[logic]['total'] += 1
                        if q['correct']:
                            logic_stats[logic]['correct'] += 1
                
                for logic in sorted(logic_stats.keys()):
                    stats = logic_stats[logic]
                    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                    f.write(f"  {logic}: {stats['correct']}/{stats['total']} ({acc:.2%})\n")
    
    def _save_logic_depth_summaries(self):
        """Save detailed breakdown by logic type AND depth"""
        summary_file = f"{self.summary_dir}/logic_depth_matrix.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Accuracy Matrix: Logic Type × Depth\n")
            f.write("=" * 70 + "\n\n")
            
            # Get all logic types and depths
            all_logics = sorted(self.results_by_logic_depth.keys())
            all_depths = set()
            for logic_results in self.results_by_logic_depth.values():
                all_depths.update(logic_results.keys())
            all_depths = sorted(all_depths)
            
            # Header
            f.write(f"{'Logic Type':<30}")
            for depth in all_depths:
                f.write(f"{depth:>12}")
            f.write("\n")
            f.write("-" * 70 + "\n")
            
            # Data rows
            for logic in all_logics:
                f.write(f"{logic:<30}")
                for depth in all_depths:
                    results = self.results_by_logic_depth[logic].get(depth, [])
                    if results:
                        total = sum(len(r['results']) for r in results)
                        correct = sum(sum(q['correct'] for q in r['results']) for r in results)
                        acc = correct / total if total > 0 else 0
                        f.write(f"{acc:>11.1%} ")
                    else:
                        f.write(f"{'N/A':>12}")
                f.write("\n")
    
    def _save_overall_summary(self):
        """Save overall summary statistics"""
        summary_file = f"{self.summary_dir}/overall_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Overall Multi-LogiEval Test Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Load all results
            with open(self.detailed_file, 'r') as df:
                all_results = json.load(df)
            
            total_files = len(all_results)
            total_questions = sum(len(r.get('results', [])) for r in all_results)
            total_correct = sum(sum(q['correct'] for q in r.get('results', [])) for r in all_results)
            
            f.write(f"Total Files Tested: {total_files}\n")
            f.write(f"Total Questions: {total_questions}\n")
            f.write(f"Total Correct: {total_correct}\n")
            if total_questions > 0:
                f.write(f"Overall Accuracy: {total_correct/total_questions:.2%}\n\n")
            
            # Summary by logic type
            f.write("\nAccuracy by Logic Type:\n")
            f.write("-" * 70 + "\n")
            for logic_type in sorted(self.results_by_logic.keys()):
                results = self.results_by_logic[logic_type]
                total_q = sum(len(r['results']) for r in results)
                correct_q = sum(sum(q['correct'] for q in r['results']) for r in results)
                acc = correct_q / total_q if total_q > 0 else 0
                f.write(f"{logic_type:<30} {correct_q:>4}/{total_q:<4} ({acc:.2%})\n")
            
            # Summary by depth
            f.write("\nAccuracy by Depth:\n")
            f.write("-" * 70 + "\n")
            for depth in sorted(self.results_by_depth.keys()):
                results = self.results_by_depth[depth]
                total_q = sum(len(r['results']) for r in results)
                correct_q = sum(sum(q['correct'] for q in r['results']) for r in results)
                acc = correct_q / total_q if total_q > 0 else 0
                f.write(f"{depth:<30} {correct_q:>4}/{total_q:<4} ({acc:.2%})\n")
    
    def finalize(self, total_questions, total_correct):
        """Write final summary and save all aggregated results"""
        try:
            # Save all aggregated summaries
            self._save_aggregated_results()
            
            # Update progress file with final results
            with open(self.progress_file, 'a') as f:
                f.write("\n" + "=" * 70 + "\n")
                f.write("FINAL RESULTS\n")
                f.write("=" * 70 + "\n")
                if total_questions > 0:
                    overall_accuracy = total_correct / total_questions
                    f.write(f"Overall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})\n")
                else:
                    f.write("No questions completed successfully.\n")
                f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"Warning: Could not finalize results: {e}")
        
        print(f"\n{'='*70}")
        print("All results saved!")
        print(f"{'='*70}")
        print(f"Main results:        {self.detailed_file}")
        print(f"Progress log:        {self.progress_file}")
        print(f"Individual responses: {self.responses_dir}/")
        print(f"By logic type:       {self.by_logic_dir}/")
        print(f"By depth:            {self.by_depth_dir}/")
        print(f"Summaries:           {self.summary_dir}/")
        print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(description='Test GPT-5 on Multi-LogiEval dataset with comprehensive result organization')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--data_dir', required=True, help='Path to Multi-LogiEval data directory')
    parser.add_argument('--logic_types', nargs='+', 
                        default=['fol', 'nm', 'pl'],
                        help='Logic types to test (fol=first-order logic, nm=non-monotonic, pl=propositional)')
    parser.add_argument('--depths', nargs='+',
                        default=['d1_Data', 'd2_Data', 'd3_Data', 'd4_Data', 'd5_Data'],
                        help='Depths to test')
    parser.add_argument('--num_files', type=int, default=0,
                        help='Number of files to test (0 = all)')
    parser.add_argument('--output_dir', default='results', help='Directory to save responses')
    parser.add_argument('--model', default='gpt-5', help='Model to use')
    args = parser.parse_args()
    
    # Load dataset
    all_files = load_multilogieval_dataset(args.data_dir, args.logic_types, args.depths)
    
    if not all_files:
        print("No files found! Check your data directory and filters.")
        return
    
    # Select files to test
    if args.num_files > 0:
        files_to_test = all_files[:args.num_files]
    else:
        files_to_test = all_files
    
    print(f"\nTesting {len(files_to_test)} files with {args.model}")
    print(f"Results will be saved to: {args.output_dir}")
    
    # Initialize incremental saver
    saver = IncrementalSaver(args.output_dir)
    
    total_questions = 0
    total_correct = 0
    
    try:
        for i, file_data in enumerate(files_to_test):
            print(f"\n{'='*70}")
            print(f"File {i+1}/{len(files_to_test)}")
            print(f"Logic: {file_data['logic_type']}, Rule: {file_data['rule']}, Depth: {file_data['depth']}")
            print(f"Samples: {len(file_data['samples'])}")
            print(f"{'='*70}")
            
            result = test_gpt5_multilogieval(file_data, args.api_key, args.model)
            
            # Save immediately after each result
            saver.save_result(result, i, len(files_to_test))
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            print(f"\n✅ File accuracy: {result['file_accuracy']:.2%}")
            for q_result in result['results']:
                status = '✓' if q_result['correct'] else '✗'
                print(f"  Q{q_result['question_num']}: {q_result['ground_truth']} → {q_result['prediction']} {status}")
                total_questions += 1
                if q_result['correct']:
                    total_correct += 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving final results...")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}. Saving results so far...")
    
    # Finalize results
    print("\n" + "="*70)
    print("Finalizing results and generating summaries...")
    print("="*70)
    saver.finalize(total_questions, total_correct)
    
    if total_questions > 0:
        overall_accuracy = total_correct / total_questions
        print(f"\n{'='*70}")
        print(f"OVERALL ACCURACY: {total_correct}/{total_questions} ({overall_accuracy:.2%})")
        print(f"{'='*70}")
    else:
        print("\n⚠️  No questions were completed successfully.")

if __name__ == "__main__":
    main()