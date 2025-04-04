#!/usr/bin/env python
"""
Generate a precision-focused LaTeX table from evaluation results.

This script generates a publication-ready LaTeX table that prioritizes precision
and throughput (chars per second) as the two most important metrics. The rows are
sorted by precision, and the columns are organized to emphasize precision first.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def load_evaluation_results(results_path: str) -> Dict[str, Any]:
    """Load evaluation results from a JSON file.
    
    Args:
        results_path: Path to the evaluation results JSON file
        
    Returns:
        Evaluation results dictionary
    """
    with open(results_path, 'r') as f:
        return json.load(f)


def calculate_chars_per_second(time_per_char_seconds: float) -> float:
    """Calculate characters per second from time per character.
    
    Args:
        time_per_char_seconds: Time per character in seconds
        
    Returns:
        Characters per second
    """
    if time_per_char_seconds <= 0:
        return float('inf')  # Avoid division by zero
    return 1.0 / time_per_char_seconds


def generate_precision_table(results: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Generate a precision-focused LaTeX table from evaluation results.
    
    Args:
        results: Evaluation results dictionary
        output_path: Optional path to save the LaTeX file
        
    Returns:
        LaTeX table code
    """
    # Prepare datasets and tokenizers
    datasets = results["datasets"]
    tokenizers = results["tokenizers"]
    
    # Prepare LaTeX header
    latex = """\\begin{table*}[htbp]
\\centering
\\caption{Sentence Boundary Detection Performance on Legal Texts (Precision-Focused)}
\\label{tab:sbd-precision-performance}
\\begin{tabular}{llrrrr}
\\toprule
Dataset & Model & \\textbf{Precision} & Chars/sec & F1 & Recall \\\\
\\midrule
"""
    
    # Tokenizer metric data
    for dataset in datasets:
        first_dataset = True
        
        # Sort tokenizers by precision score for this dataset
        sorted_tokenizers = sorted(
            tokenizers,
            key=lambda t: results["results"][t][dataset]["summary"]["precision"],
            reverse=True
        )
        
        for tokenizer in sorted_tokenizers:
            summary = results["results"][tokenizer][dataset]["summary"]
            
            # Format dataset name (only for first row)
            if first_dataset:
                dataset_name = dataset.replace("multilegal_", "")
                dataset_name = dataset_name.replace("_", " ").title()
                first_dataset = False
            else:
                dataset_name = ""
            
            # Calculate characters per second (inverse of time_per_char_seconds)
            chars_per_sec = calculate_chars_per_second(summary["time_per_char_seconds"])
            
            # Format chars_per_sec with appropriate units and precision
            if chars_per_sec > 1000000:
                chars_per_sec_formatted = f"{chars_per_sec/1000000:.2f}M"
            elif chars_per_sec > 1000:
                chars_per_sec_formatted = f"{chars_per_sec/1000:.2f}K"
            elif chars_per_sec == float('inf'):
                chars_per_sec_formatted = ">1M"
            else:
                chars_per_sec_formatted = f"{chars_per_sec:.1f}"
            
            # Add row
            latex += f"{dataset_name} & {tokenizer} & {summary['precision']:.3f} & {chars_per_sec_formatted} & {summary['f1']:.3f} & {summary['recall']:.3f} \\\\\n"
        
        # Add a midrule between datasets
        if dataset != datasets[-1]:
            latex += "\\midrule\n"
    
    # Close the LaTeX table
    latex += """\\bottomrule
\\end{tabular}
\\end{table*}
"""
    
    # Save if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"Precision-focused LaTeX table saved to {output_path}")
    
    return latex


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate a precision-focused LaTeX table from evaluation results")
    parser.add_argument("results_file", type=str, help="Path to evaluation results JSON file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for LaTeX table")
    args = parser.parse_args()
    
    # Load evaluation results
    try:
        results = load_evaluation_results(args.results_file)
    except Exception as e:
        print(f"Error loading evaluation results: {e}")
        sys.exit(1)
    
    # Generate and print table
    latex_table = generate_precision_table(results, args.output)
    if not args.output:
        print(latex_table)


if __name__ == "__main__":
    main()