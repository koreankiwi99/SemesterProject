"""
Generate Multi-LogiEval samples using GPT-4o following the paper's methodology.
Reusable for all combinations and depths.
"""

import json
import argparse
from pathlib import Path
from openai import OpenAI

COMBINATION_RULES = {
    1: "HS_CD_DS_MP_HS_MP_MP",
    2: "BD_CT_DS_HS_MP_MP_MP",
    3: "CD_DS_HS_CD_DS_MP_MP",
    4: "HS_MT_DS_BD_CT_DS_MP",
    5: "DD_DS_HS_MT_DS_MP_MP"
}


def load_system_prompt() -> str:
    """Load the system prompt from file."""
    prompt_dir = Path(__file__).parent.parent.parent / "prompts" / "multilogi" / "generation"
    system_prompt_file = prompt_dir / "system_prompt.txt"

    with open(system_prompt_file, 'r') as f:
        return f.read().strip()


def load_prompt(combination: int, depth: str) -> str:
    """Load the user prompt file for the given combination and depth."""
    prompt_dir = Path(__file__).parent.parent.parent / "prompts" / "multilogi" / "generation" / "pl"
    prompt_file = prompt_dir / f"combination_{combination}_{depth}.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, 'r') as f:
        return f.read().strip()


def generate_samples(combination: int, depth: str, temperature: float = 0.7,
                     model: str = "gpt-4o") -> str:
    """Generate samples using GPT-4o."""
    client = OpenAI()
    system_prompt = load_system_prompt()
    user_prompt = load_prompt(combination, depth)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )

    return response.choices[0].message.content


def save_output(response_text: str, combination: int, depth: str, output_dir: Path):
    """Save raw output and metadata."""
    raw_file = output_dir / f"combination_{combination}_{depth}_raw.txt"
    with open(raw_file, 'w') as f:
        f.write(response_text)

    meta_file = output_dir / f"combination_{combination}_{depth}_meta.json"
    metadata = {
        "logic": "pl",
        "rule": COMBINATION_RULES.get(combination, f"combination_{combination}"),
        "depth": depth,
        "combination": combination,
        "raw_output_file": str(raw_file.name)
    }

    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Raw output saved to: {raw_file}")
    print(f"Metadata saved to: {meta_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Multi-LogiEval samples using GPT-4o"
    )
    parser.add_argument(
        "--combination", type=int, required=True, choices=[1, 2, 3, 4, 5],
        help="Combination number (1-5)"
    )
    parser.add_argument(
        "--depth", type=str, required=True,
        help="Depth identifier (e.g., 'd7', 'd10')"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="Model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/generated/multilogi",
        help="Output directory (default: data/generated/multilogi)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating samples for Combination {args.combination}, Depth {args.depth}")
    print(f"Model: {args.model}, Temperature: {args.temperature}")

    response_text = generate_samples(
        combination=args.combination,
        depth=args.depth,
        temperature=args.temperature,
        model=args.model
    )

    save_output(response_text, args.combination, args.depth, output_dir)
    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
