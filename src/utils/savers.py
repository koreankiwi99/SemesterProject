"""Result saving utilities."""

import json
import os
from datetime import datetime
from collections import defaultdict


class BaseSaver:
    """Base class for saving experiment results."""

    def __init__(self, output_dir="results", experiment_name="test"):
        """Initialize saver with output directory and experiment name.

        Args:
            output_dir: Base output directory
            experiment_name: Name of the experiment for file naming
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)

    def _append_to_json(self, file_path, result):
        """Append a result to a JSON file.

        Args:
            file_path: Path to the JSON file
            result: Result dictionary to append
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            data.append(result)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update JSON file: {e}")


class MultiLogiEvalSaver(BaseSaver):
    """Saver for Multi-LogiEval experiment results."""

    def __init__(self, output_dir="results", prompt_name="test"):
        super().__init__(output_dir, prompt_name)

        self.base_dir = f"{output_dir}/multilogieval_{prompt_name}_{self.timestamp}"
        os.makedirs(self.base_dir, exist_ok=True)

        self.all_results_file = f"{self.base_dir}/all_results.json"
        self.summary_file = f"{self.base_dir}/summary.txt"
        self.accuracy_table_file = f"{self.base_dir}/accuracy_table.txt"
        self.progress_file = f"{self.base_dir}/progress.txt"

        self._init_files()

    def _init_files(self):
        """Initialize output files."""
        with open(self.all_results_file, 'w') as f:
            json.dump([], f)

        with open(self.progress_file, 'w') as f:
            f.write(f"Multi-LogiEval Testing - Started at {self.timestamp}\n")
            f.write("=" * 70 + "\n\n")

    def save_result(self, result, question_index, total_questions):
        """Save a single result incrementally."""
        self._append_to_json(self.all_results_file, result)
        self._update_progress(result, question_index, total_questions)

    def _update_progress(self, result, question_index, total_questions):
        """Update progress file."""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"Question {question_index+1}/{total_questions}\n")
                f.write(f"  Logic: {result.get('logic_type')}, Depth: {result.get('depth_dir')}, Rule: {result.get('rule')}\n")

                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    f.write(f"  Result: {result['ground_truth']} → {result['prediction']} "
                           f"{'✓' if result['correct'] else '✗'}\n")

                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")

    def finalize(self, results, results_by_combination):
        """Generate final summaries."""
        self._save_summary(results)
        self._save_accuracy_table(results_by_combination)

        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"All results:    {self.all_results_file}")
        print(f"Summary:        {self.summary_file}")
        print(f"Accuracy table: {self.accuracy_table_file}")
        print(f"Progress log:   {self.progress_file}")
        print(f"{'='*70}")

    def _save_summary(self, results):
        """Save overall summary."""
        with open(self.summary_file, 'w') as f:
            f.write("Multi-LogiEval Evaluation Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            total_correct = sum(r['correct'] for r in results if 'error' not in r)
            total_questions = len([r for r in results if 'error' not in r])

            if total_questions > 0:
                overall_accuracy = total_correct / total_questions
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n\n")
            else:
                f.write("No questions completed successfully.\n\n")

            # Accuracy by logic type
            f.write("Accuracy by Logic Type:\n")
            f.write("-" * 70 + "\n")
            logic_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
            for r in results:
                if 'error' not in r:
                    logic_stats[r['logic_type']]['total'] += 1
                    if r['correct']:
                        logic_stats[r['logic_type']]['correct'] += 1

            for logic_type in sorted(logic_stats.keys()):
                stats = logic_stats[logic_type]
                if stats['total'] > 0:
                    acc = stats['correct'] / stats['total']
                    f.write(f"{logic_type.upper():<10} {stats['correct']:>3}/{stats['total']:<3} ({acc:.2%})\n")

    def _save_accuracy_table(self, results_by_combination):
        """Save accuracy table in paper's Table 6 format."""
        with open(self.accuracy_table_file, 'w') as f:
            f.write("Accuracy Table (Paper's Table 6 Format)\n")
            f.write("=" * 70 + "\n\n")

            logic_types = ['pl', 'fol', 'nm']
            depths = ['d1_Data', 'd2_Data', 'd3_Data', 'd4_Data', 'd5_Data']

            # Calculate accuracies
            accuracies = {}
            for logic_type in logic_types:
                accuracies[logic_type] = {}
                for depth in depths:
                    key = (logic_type, depth)
                    if key in results_by_combination:
                        results = [r for r in results_by_combination[key] if 'error' not in r]
                        if results:
                            correct = sum(r['correct'] for r in results)
                            total = len(results)
                            accuracies[logic_type][depth] = correct / total * 100
                        else:
                            accuracies[logic_type][depth] = None
                    else:
                        accuracies[logic_type][depth] = None

            # Print table
            f.write(f"{'Logic':<15}")
            for depth in depths:
                f.write(f"{depth.replace('_Data', ''):>10}")
            f.write(f"{'Average':>12}\n")
            f.write("-" * 70 + "\n")

            # Print rows
            for logic_type in logic_types:
                f.write(f"{logic_type.upper():<15}")
                valid_accs = []
                for depth in depths:
                    acc = accuracies[logic_type][depth]
                    if acc is not None:
                        f.write(f"{acc:>9.2f}%")
                        valid_accs.append(acc)
                    else:
                        f.write(f"{'N/A':>10}")

                if valid_accs:
                    avg = sum(valid_accs) / len(valid_accs)
                    f.write(f"{avg:>11.2f}%")
                else:
                    f.write(f"{'N/A':>12}")
                f.write("\n")


class MultiLogiEvalLeanSaver(MultiLogiEvalSaver):
    """Saver for Multi-LogiEval experiments with Lean verification."""

    def __init__(self, output_dir="results", prompt_name="lean_test"):
        # Initialize base class but don't create parent directories/files
        BaseSaver.__init__(self, output_dir, prompt_name)

        # Create Lean-specific directory structure
        self.base_dir = f"{output_dir}/multilogieval_lean_{prompt_name}_{self.timestamp}"
        os.makedirs(self.base_dir, exist_ok=True)

        self.responses_dir = f"{self.base_dir}/responses"
        os.makedirs(self.responses_dir, exist_ok=True)

        self.all_results_file = f"{self.base_dir}/all_results.json"
        self.summary_file = f"{self.base_dir}/summary.txt"
        self.accuracy_table_file = f"{self.base_dir}/accuracy_table.txt"
        self.lean_stats_file = f"{self.base_dir}/lean_verification_stats.txt"
        self.progress_file = f"{self.base_dir}/progress.txt"

        self._init_files()

    def _init_files(self):
        """Initialize output files."""
        with open(self.all_results_file, 'w') as f:
            json.dump([], f)

        with open(self.progress_file, 'w') as f:
            f.write(f"Multi-LogiEval + Lean - Started at {self.timestamp}\n")
            f.write("=" * 70 + "\n\n")

    def save_result(self, result, question_index, total_questions):
        """Save a single result incrementally."""
        self._append_to_json(self.all_results_file, result)

        if 'error' not in result:
            self._save_individual_response(result, question_index)

        self._update_progress_lean(result, question_index, total_questions)

    def _save_individual_response(self, result, question_index):
        """Save individual response file."""
        filename = f"q{question_index + 1:03d}_{result['logic_type']}_{result['depth_dir']}_{result['rule']}.txt"
        response_file = f"{self.responses_dir}/{filename}"

        try:
            with open(response_file, 'w') as f:
                f.write("Multi-LogiEval Question with Lean Verification\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Logic Type: {result['logic_type']}\n")
                f.write(f"Depth: {result['depth']} ({result['depth_dir']})\n")
                f.write(f"Rule: {result['rule']}\n\n")

                f.write("Context:\n")
                f.write(result['context'] + "\n\n")

                f.write("Question:\n")
                f.write(result['question'] + "\n\n")

                f.write("=" * 80 + "\n")
                f.write("Iterations:\n")
                f.write("=" * 80 + "\n\n")

                for iter_data in result.get('iterations', []):
                    f.write(f"--- Iteration {iter_data['iteration']} ---\n\n")
                    f.write("LLM Response:\n")
                    f.write(iter_data['llm_response'] + "\n\n")

                    if iter_data.get('lean_code'):
                        f.write("Extracted Lean Code:\n")
                        f.write("-" * 40 + "\n")
                        f.write(iter_data['lean_code'] + "\n")
                        f.write("-" * 40 + "\n\n")

                        if iter_data.get('lean_verification'):
                            verification = iter_data['lean_verification']
                            f.write("Lean Verification:\n")
                            f.write(f"  Success: {verification['success']}\n")
                            if verification['errors']:
                                f.write("  Errors:\n")
                                for err in verification['errors']:
                                    f.write(f"    - {err}\n")
                    else:
                        f.write("No Lean code found.\n")

                    f.write("\n")

                f.write("=" * 80 + "\n")
                f.write("Final Result:\n")
                f.write("=" * 80 + "\n")
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Prediction: {result['prediction']}\n")
                f.write(f"Correct: {'✓ Yes' if result['correct'] else '✗ No'}\n")
                f.write(f"Total Iterations: {result['num_iterations']}\n")

                if result.get('lean_verification'):
                    f.write(f"Final Lean: {'✓ Success' if result['lean_verification']['success'] else '✗ Failed'}\n")

        except Exception as e:
            print(f"Warning: Could not save individual response: {e}")

    def _update_progress_lean(self, result, question_index, total_questions):
        """Update progress file with Lean info."""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"Question {question_index+1}/{total_questions}\n")
                f.write(f"  Logic: {result.get('logic_type')}, Depth: {result.get('depth_dir')}, Rule: {result.get('rule')}\n")

                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    f.write(f"  Result: {result['ground_truth']} → {result['prediction']} "
                           f"{'✓' if result['correct'] else '✗'}\n")
                    f.write(f"  Iterations: {result['num_iterations']}\n")

                    if result.get('lean_verification'):
                        lean_success = result['lean_verification']['success']
                        f.write(f"  Lean: {'Success' if lean_success else 'Failed'}\n")
                    elif result.get('lean_code') is None:
                        f.write("  Lean: No code generated\n")

                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")

    def finalize(self, results, results_by_combination):
        """Generate final summaries including Lean stats."""
        self._save_summary(results)
        self._save_accuracy_table(results_by_combination)
        self._save_lean_stats(results)

        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"All results:    {self.all_results_file}")
        print(f"Summary:        {self.summary_file}")
        print(f"Accuracy table: {self.accuracy_table_file}")
        print(f"Lean stats:     {self.lean_stats_file}")
        print(f"Progress log:   {self.progress_file}")
        print(f"Responses:      {self.responses_dir}/")
        print(f"{'='*70}")

    def _save_lean_stats(self, results):
        """Save Lean verification statistics."""
        with open(self.lean_stats_file, 'w') as f:
            f.write("Lean Verification Statistics\n")
            f.write("=" * 70 + "\n\n")

            total_questions = 0
            questions_with_code = 0
            successful_verifications = 0
            failed_verifications = 0
            total_iterations = 0

            iteration_counts = defaultdict(int)

            for r in results:
                if 'error' not in r:
                    total_questions += 1
                    total_iterations += r['num_iterations']
                    iteration_counts[r['num_iterations']] += 1

                    if r.get('lean_code'):
                        questions_with_code += 1
                        if r.get('lean_verification'):
                            if r['lean_verification']['success']:
                                successful_verifications += 1
                            else:
                                failed_verifications += 1

            f.write(f"Total Questions: {total_questions}\n")
            f.write(f"Questions with Lean code: {questions_with_code}\n")
            f.write(f"Successful verifications: {successful_verifications}\n")
            f.write(f"Failed verifications: {failed_verifications}\n")

            if total_questions > 0:
                avg_iterations = total_iterations / total_questions
                f.write(f"\nAverage iterations: {avg_iterations:.2f}\n")

            if questions_with_code > 0:
                verification_rate = successful_verifications / questions_with_code
                f.write(f"Verification success rate: {verification_rate:.2%}\n")

            f.write("\nIteration Distribution:\n")
            for num_iter in sorted(iteration_counts.keys()):
                count = iteration_counts[num_iter]
                f.write(f"  {num_iter} iteration(s): {count} questions\n")


class FOLIOSaver(BaseSaver):
    """Saver for FOLIO experiment results."""

    def __init__(self, output_dir="results", prompt_name="test"):
        super().__init__(output_dir, prompt_name)

        self.base_dir = f"{output_dir}/folio_{prompt_name}_{self.timestamp}"
        os.makedirs(self.base_dir, exist_ok=True)

        self.all_results_file = f"{self.base_dir}/all_results.json"
        self.responses_dir = f"{self.base_dir}/responses"
        self.progress_file = f"{self.base_dir}/progress.txt"
        self.summary_file = f"{self.base_dir}/summary.txt"

        os.makedirs(self.responses_dir, exist_ok=True)
        self._init_files()

    def _init_files(self):
        """Initialize output files."""
        with open(self.all_results_file, 'w') as f:
            json.dump([], f)

        with open(self.progress_file, 'w') as f:
            f.write(f"FOLIO Testing - Started at {self.timestamp}\n")
            f.write("=" * 70 + "\n\n")

    def save_result(self, result, story_index, total_stories):
        """Save a single story result."""
        self._append_to_json(self.all_results_file, result)

        if 'error' not in result:
            self._save_individual_response(result)

        self._update_progress(result, story_index, total_stories)
        print(f"✓ Saved result for story {result.get('story_id', 'unknown')}")

    def _save_individual_response(self, result):
        """Save individual response file."""
        story_id = result['story_id']
        response_file = f"{self.responses_dir}/story_{story_id}_response.txt"

        try:
            with open(response_file, 'w') as f:
                f.write(f"Story ID: {story_id}\n")
                f.write(f"Premises: {result['premises']}\n\n")
                f.write("=" * 50 + "\n")
                f.write("Model Response:\n")
                f.write("=" * 50 + "\n")
                f.write(result['model_response'])
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
        """Update progress file."""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"Story {story_index+1}/{total_stories}: {result.get('story_id', 'unknown')}\n")
                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    accuracy = result['story_accuracy']
                    num_questions = len(result['results'])
                    correct_count = sum(r['correct'] for r in result['results'])
                    f.write(f"  Accuracy: {correct_count}/{num_questions} ({accuracy:.2%})\n")
                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")

    def finalize(self, total_questions, total_correct):
        """Write final summary."""
        # Save to summary file
        with open(self.summary_file, 'w') as f:
            f.write("FOLIO Evaluation Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if total_questions > 0:
                overall_accuracy = total_correct / total_questions
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n")
            else:
                f.write("No questions completed successfully.\n")

        # Also update progress file
        try:
            with open(self.progress_file, 'a') as f:
                f.write("=" * 70 + "\n")
                f.write("FINAL RESULTS\n")
                f.write("=" * 70 + "\n")
                if total_questions > 0:
                    overall_accuracy = total_correct / total_questions
                    f.write(f"Overall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})\n")
                else:
                    f.write("No questions completed successfully.\n")
                f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"Warning: Could not finalize progress: {e}")

        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"All results:    {self.all_results_file}")
        print(f"Summary:        {self.summary_file}")
        print(f"Progress log:   {self.progress_file}")
        print(f"Responses:      {self.responses_dir}/")
        print(f"{'='*70}")


class FOLIOLeanSaver(BaseSaver):
    """Saver for FOLIO experiments with Lean verification."""

    def __init__(self, output_dir="results", prompt_name="lean_test"):
        super().__init__(output_dir, prompt_name)

        self.base_dir = f"{output_dir}/folio_lean_{prompt_name}_{self.timestamp}"
        os.makedirs(self.base_dir, exist_ok=True)

        self.all_results_file = f"{self.base_dir}/all_results.json"
        self.responses_dir = f"{self.base_dir}/responses"
        self.progress_file = f"{self.base_dir}/progress.txt"
        self.summary_file = f"{self.base_dir}/summary.txt"
        self.lean_stats_file = f"{self.base_dir}/lean_verification_stats.txt"

        os.makedirs(self.responses_dir, exist_ok=True)
        self._init_files()

    def _init_files(self):
        """Initialize output files."""
        with open(self.all_results_file, 'w') as f:
            json.dump([], f)

        with open(self.progress_file, 'w') as f:
            f.write(f"FOLIO + Lean - Started at {self.timestamp}\n")
            f.write("=" * 70 + "\n\n")

    def save_result(self, result, question_index, total_questions):
        """Save a single question result."""
        self._append_to_json(self.all_results_file, result)

        if 'error' not in result:
            self._save_individual_response(result)

        self._update_progress(result, question_index, total_questions)

    def _save_individual_response(self, result):
        """Save individual response file."""
        story_id = result['story_id']
        example_id = result['example_id']
        response_file = f"{self.responses_dir}/story_{story_id}_q{example_id}.txt"

        try:
            with open(response_file, 'w') as f:
                f.write(f"Story ID: {story_id}\n")
                f.write(f"Example ID: {example_id}\n")
                f.write(f"Premises: {result['premises']}\n")
                f.write(f"Conclusion: {result['conclusion']}\n\n")

                # Write all iterations
                for iter_data in result.get('iterations', []):
                    f.write("=" * 50 + "\n")
                    f.write(f"Iteration {iter_data['iteration']}\n")
                    f.write("=" * 50 + "\n")
                    f.write(iter_data['llm_response'])
                    f.write("\n\n")

                    if iter_data.get('lean_code'):
                        f.write("--- Lean Code ---\n")
                        f.write(iter_data['lean_code'])
                        f.write("\n\n")

                    if iter_data.get('lean_verification'):
                        f.write("--- Lean Verification ---\n")
                        verification = iter_data['lean_verification']
                        f.write(f"Success: {verification['success']}\n")
                        if verification['errors']:
                            f.write(f"Errors:\n")
                            for err in verification['errors']:
                                f.write(f"  - {err}\n")
                        if verification['warnings']:
                            f.write(f"Warnings:\n")
                            for warn in verification['warnings']:
                                f.write(f"  - {warn}\n")
                        f.write("\n")

                f.write("=" * 50 + "\n")
                f.write("Final Result:\n")
                f.write("=" * 50 + "\n")
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Prediction: {result['prediction']}\n")
                f.write(f"Correct: {'Yes' if result['correct'] else 'No'}\n")
                f.write(f"Total Iterations: {result['num_iterations']}\n")
        except Exception as e:
            print(f"Warning: Could not save individual response: {e}")

    def _update_progress(self, result, question_index, total_questions):
        """Update progress file."""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"Question {question_index+1}/{total_questions}\n")
                f.write(f"  Story: {result.get('story_id')}, Example: {result.get('example_id')}\n")

                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    f.write(f"  Result: {result['ground_truth']} → {result['prediction']} "
                           f"{'✓' if result['correct'] else '✗'}\n")
                    f.write(f"  Iterations: {result['num_iterations']}\n")

                    if result.get('lean_verification'):
                        lean_success = result['lean_verification']['success']
                        f.write(f"  Lean: {'Success' if lean_success else 'Failed'}\n")
                    elif result.get('lean_code') is None:
                        f.write("  Lean: No code found\n")

                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")

    def finalize(self, total_questions, total_correct, lean_stats):
        """Write final summary."""
        # Save to summary file
        with open(self.summary_file, 'w') as f:
            f.write("FOLIO + Lean Evaluation Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if total_questions > 0:
                overall_accuracy = total_correct / total_questions
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n\n")
            else:
                f.write("No questions completed successfully.\n\n")

            f.write("Lean Verification Stats:\n")
            f.write(f"  Questions with Lean code: {lean_stats['with_code']}\n")
            f.write(f"  Successful verifications: {lean_stats['successful']}\n")
            f.write(f"  Failed verifications: {lean_stats['failed']}\n")
            f.write(f"  Average iterations: {lean_stats['avg_iterations']:.2f}\n")
            if lean_stats['with_code'] > 0:
                verification_rate = lean_stats['successful'] / lean_stats['with_code']
                f.write(f"  Verification success rate: {verification_rate:.2%}\n")

        # Also update progress file
        try:
            with open(self.progress_file, 'a') as f:
                f.write("=" * 70 + "\n")
                f.write("FINAL RESULTS\n")
                f.write("=" * 70 + "\n")
                if total_questions > 0:
                    overall_accuracy = total_correct / total_questions
                    f.write(f"Overall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})\n")
                else:
                    f.write("No questions completed successfully.\n")

                f.write(f"\nLean Verification Stats:\n")
                f.write(f"  Questions with Lean code: {lean_stats['with_code']}\n")
                f.write(f"  Successful verifications: {lean_stats['successful']}\n")
                f.write(f"  Failed verifications: {lean_stats['failed']}\n")
                f.write(f"  Average iterations: {lean_stats['avg_iterations']:.2f}\n")
                if lean_stats['with_code'] > 0:
                    verification_rate = lean_stats['successful'] / lean_stats['with_code']
                    f.write(f"  Verification success rate: {verification_rate:.2%}\n")

                f.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"Warning: Could not finalize progress: {e}")

        # Save detailed Lean stats
        self._save_lean_stats(lean_stats)

        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"All results:    {self.all_results_file}")
        print(f"Summary:        {self.summary_file}")
        print(f"Lean stats:     {self.lean_stats_file}")
        print(f"Progress log:   {self.progress_file}")
        print(f"Responses:      {self.responses_dir}/")
        print(f"{'='*70}")

    def _save_lean_stats(self, lean_stats):
        """Save detailed Lean statistics."""
        with open(self.lean_stats_file, 'w') as f:
            f.write("Lean Verification Statistics\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Questions with Lean code: {lean_stats['with_code']}\n")
            f.write(f"Successful verifications: {lean_stats['successful']}\n")
            f.write(f"Failed verifications: {lean_stats['failed']}\n")
            f.write(f"Average iterations: {lean_stats['avg_iterations']:.2f}\n")

            if lean_stats['with_code'] > 0:
                verification_rate = lean_stats['successful'] / lean_stats['with_code']
                f.write(f"Verification success rate: {verification_rate:.2%}\n")
