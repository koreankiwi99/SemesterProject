#!/usr/bin/env python3
"""
Gather all experiment results and produce comprehensive summary tables.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path("/Users/kyuheekim/SemesterProject")


def load_json(path):
    """Load JSON file safely."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def analyze_by_depth_and_type(results):
    """Analyze results by depth and logic type."""
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        depth = r.get("depth", r.get("depth_dir", "unknown"))
        if "_Data" in str(depth):
            depth = depth.replace("_Data", "")
        logic = r.get("logic_type", "unknown")
        correct = r.get("correct", False)

        # By depth
        stats[f"depth_{depth}"]["total"] += 1
        if correct:
            stats[f"depth_{depth}"]["correct"] += 1

        # By type
        stats[f"type_{logic}"]["total"] += 1
        if correct:
            stats[f"type_{logic}"]["correct"] += 1

        # By depth x type
        stats[f"{depth}_{logic}"]["total"] += 1
        if correct:
            stats[f"{depth}_{logic}"]["correct"] += 1

        # Overall
        stats["overall"]["total"] += 1
        if correct:
            stats["overall"]["correct"] += 1

    return stats


def format_accuracy(stats, key):
    """Format accuracy as percentage."""
    if key not in stats or stats[key]["total"] == 0:
        return "-"
    s = stats[key]
    return f"{s['correct']}/{s['total']} ({100*s['correct']/s['total']:.1f}%)"


def print_multilogieval_table(name, stats):
    """Print formatted table for Multi-LogiEval results."""
    print(f"\n### {name}")
    print(f"Overall: {format_accuracy(stats, 'overall')}")

    print("\n**By Depth:**")
    print("| Depth | Accuracy |")
    print("|-------|----------|")
    for d in ["d1", "d2", "d3", "d4", "d5", "d7"]:
        acc = format_accuracy(stats, f"depth_{d}")
        if acc != "-":
            print(f"| {d} | {acc} |")

    print("\n**By Logic Type:**")
    print("| Type | Accuracy |")
    print("|------|----------|")
    for t in ["fol", "nm", "pl"]:
        acc = format_accuracy(stats, f"type_{t}")
        if acc != "-":
            print(f"| {t.upper()} | {acc} |")

    print("\n**By Depth × Type:**")
    print("| Depth | FOL | NM | PL |")
    print("|-------|-----|----|----|")
    for d in ["d1", "d2", "d3", "d4", "d5"]:
        fol = format_accuracy(stats, f"{d}_fol")
        nm = format_accuracy(stats, f"{d}_nm")
        pl = format_accuracy(stats, f"{d}_pl")
        if fol != "-" or nm != "-" or pl != "-":
            print(f"| {d} | {fol} | {nm} | {pl} |")


def analyze_folio(results):
    """Analyze FOLIO results by label."""
    stats = {"True": {"correct": 0, "total": 0},
             "False": {"correct": 0, "total": 0},
             "Unknown": {"correct": 0, "total": 0},
             "overall": {"correct": 0, "total": 0}}

    for r in results:
        # Handle nested results structure (CoT format)
        if "results" in r:
            for sub_r in r["results"]:
                label = sub_r.get("label", sub_r.get("ground_truth", "unknown"))
                if label in ["Uncertain", "uncertain"]:
                    label = "Unknown"
                correct = sub_r.get("correct", False)

                if label in stats:
                    stats[label]["total"] += 1
                    if correct:
                        stats[label]["correct"] += 1

                stats["overall"]["total"] += 1
                if correct:
                    stats["overall"]["correct"] += 1
        else:
            # Direct results structure (Lean/Bidirectional format)
            label = r.get("label", r.get("ground_truth", "unknown"))
            if label in ["Uncertain", "uncertain"]:
                label = "Unknown"
            correct = r.get("correct", False)

            if label in stats:
                stats[label]["total"] += 1
                if correct:
                    stats[label]["correct"] += 1

            stats["overall"]["total"] += 1
            if correct:
                stats["overall"]["correct"] += 1

    return stats


def print_folio_table(name, stats):
    """Print formatted table for FOLIO results."""
    print(f"\n### {name}")
    print(f"Overall: {format_accuracy(stats, 'overall')}")

    print("\n**By Ground Truth Label:**")
    print("| Label | Accuracy |")
    print("|-------|----------|")
    for label in ["True", "False", "Unknown"]:
        print(f"| {label} | {format_accuracy(stats, label)} |")


def main():
    print("# GPT-5 Experiment Results Summary")
    print("=" * 60)

    # Multi-LogiEval results
    print("\n## Multi-LogiEval (All Depths d1-d5)")

    ml_results = {
        "CoT": BASE_DIR / "results/multilogieval/all_depths/cot/all_results.json",
        "Lean": BASE_DIR / "results/multilogieval/all_depths/lean/all_results.json",
        "Bidirectional": BASE_DIR / "results/multilogieval/all_depths/bidirectional/all_results.json",
        "Two-Stage": BASE_DIR / "results/multilogieval/all_depths/two_stage/all_results.json",
    }

    for name, path in ml_results.items():
        data = load_json(path)
        if data:
            stats = analyze_by_depth_and_type(data)
            print_multilogieval_table(name, stats)

    # Multi-LogiEval d5 only
    print("\n## Multi-LogiEval (Depth-5 Only)")

    ml_d5_results = {
        "CoT": BASE_DIR / "results/multilogieval/d5_only/cot/all_results.json",
        "Lean": BASE_DIR / "results/multilogieval/d5_only/lean/all_results.json",
        "Bidirectional": BASE_DIR / "results/multilogieval/d5_only/bidirectional/all_results.json",
    }

    for name, path in ml_d5_results.items():
        data = load_json(path)
        if data:
            stats = analyze_by_depth_and_type(data)
            print_multilogieval_table(name, stats)

    # FOLIO results
    print("\n## FOLIO Dataset")

    folio_results = {
        "CoT": BASE_DIR / "results/folio/cot/all_results.json",
        "Lean": BASE_DIR / "results/folio/lean/all_results.json",
        "Bidirectional": BASE_DIR / "results/folio/bidirectional/all_results.json",
        "Two-Stage": BASE_DIR / "results/folio/two_stage/all_results.json",
    }

    for name, path in folio_results.items():
        data = load_json(path)
        if data:
            stats = analyze_folio(data)
            print_folio_table(name, stats)

    # Simple/No-CoT results
    print("\n## GPT-5 Without Chain-of-Thought (Simple Prompts)")

    simple_results = {
        "FOLIO Simple": BASE_DIR / "results/archive/folio_simple_20251205_220912/all_results.json",
        "Multi-LogiEval Simple": BASE_DIR / "results/archive/multilogieval_simple_20251205_221137/all_results.json",
    }

    for name, path in simple_results.items():
        data = load_json(path)
        if data:
            if "FOLIO" in name:
                stats = analyze_folio(data)
                print_folio_table(name, stats)
            else:
                stats = analyze_by_depth_and_type(data)
                print_multilogieval_table(name, stats)

    # GPT-5 with reasoning_effort variations
    print("\n## GPT-5 with reasoning_effort Configurations")

    reasoning_effort_results = {
        "FOLIO CoT (minimal reasoning)": BASE_DIR / "results/archive/folio_cot_minimal_reasoning_20251205_221312/all_results.json",
        "Multi-LogiEval CoT (minimal reasoning)": BASE_DIR / "results/archive/multilogieval_cot_minimal_reasoning_20251205_221348/all_results.json",
        "FOLIO Simple (minimal reasoning)": BASE_DIR / "results/archive/folio_simple_minimal_reasoning_20251206_115637/all_results.json",
        "Multi-LogiEval Simple (minimal reasoning)": BASE_DIR / "results/archive/multilogieval_simple_minimal_reasoning_20251206_115713/all_results.json",
    }

    for name, path in reasoning_effort_results.items():
        data = load_json(path)
        if data:
            if "FOLIO" in name:
                stats = analyze_folio(data)
                print_folio_table(name, stats)
            else:
                stats = analyze_by_depth_and_type(data)
                print_multilogieval_table(name, stats)

    # Other models CoT
    print("\n## Other Models (CoT Only)")

    other_models = {
        "DeepSeek-R1 FOLIO": BASE_DIR / "results/cot/folio/deepseek_r1/all_results.json",
        "DeepSeek-R1 Multi-LogiEval": BASE_DIR / "results/cot/multilogieval/deepseek_r1/all_results.json",
        "Mistral Large FOLIO": BASE_DIR / "results/cot/folio/mistral_large/all_results.json",
        "Mistral Large Multi-LogiEval": BASE_DIR / "results/cot/multilogieval/mistral_large/all_results.json",
        "Qwen3-235B FOLIO": BASE_DIR / "results/cot/folio/qwen3_235b/all_results.json",
        "Qwen3-235B Multi-LogiEval": BASE_DIR / "results/cot/multilogieval/qwen3_235b/all_results.json",
    }

    for name, path in other_models.items():
        data = load_json(path)
        if data:
            if "FOLIO" in name:
                stats = analyze_folio(data)
                print_folio_table(name, stats)
            else:
                stats = analyze_by_depth_and_type(data)
                print_multilogieval_table(name, stats)


if __name__ == "__main__":
    main()
