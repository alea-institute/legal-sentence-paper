#!/usr/bin/env python3
"""
Generate a table with aggregate metrics (precision/recall/F1) across all datasets.

This script calculates metrics based on the combined confusion matrix across all datasets,
following the standard approach in academic publications where metrics are calculated
from raw counts rather than averaged.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# Model name mapping for cleaner publication names
MODEL_DISPLAY_NAMES = {
    'nltk_punkt': 'NLTK Punkt',
    'spacy_sm': 'spaCy (sm)',
    'spacy_lg': 'spaCy (lg)',
    'pysbd': 'PySBD',
    'nupunkt': 'NUPunkt',
    'charboundary_small': 'CharBoundary (S)',
    'charboundary_medium': 'CharBoundary (M)',
    'charboundary_large': 'CharBoundary (L)'
}


def calculate_aggregate_metrics(results):
    """Calculate aggregate metrics across all datasets from raw counts.
    
    Args:
        results: Evaluation results dictionary
        
    Returns:
        Dictionary with aggregate metrics for each tokenizer
    """
    tokenizers = results["tokenizers"]
    datasets = results["datasets"]
    
    aggregate_metrics = {}
    
    for tokenizer_name in tokenizers:
        # Initialize combined counts
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_chars = 0
        total_time = 0
        
        for dataset_name in datasets:
            result = results["results"][tokenizer_name][dataset_name]
            summary = result["summary"]
            
            # Add to total counts - accounting for different naming conventions
            if "total_true_positives" in summary and "total_false_positives" in summary and "total_false_negatives" in summary:
                total_tp += summary["total_true_positives"]
                total_fp += summary["total_false_positives"] 
                total_fn += summary["total_false_negatives"]
            elif "true_positives" in summary and "false_positives" in summary and "false_negatives" in summary:
                total_tp += summary["true_positives"]
                total_fp += summary["false_positives"]
                total_fn += summary["false_negatives"]
                
            # For timing data
            if "total_chars" in summary:
                total_chars += summary["total_chars"]
                
                if "time_per_char_seconds" in summary and summary["time_per_char_seconds"] > 0:
                    # Use consistent approach to calculate time
                    char_count = summary["total_chars"]
                    char_time = summary["time_per_char_seconds"]
                    total_time += char_count * char_time
        
        # Calculate metrics from combined counts
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate time per character in milliseconds
        time_per_char_ms = (total_time / total_chars) * 1000 if total_chars > 0 and total_time > 0 else 0
        
        # Store in results
        aggregate_metrics[tokenizer_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "time_per_char_ms": time_per_char_ms,
            "raw_counts": {
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
                "total_chars": total_chars,
                "total_time": total_time
            }
        }
    
    return aggregate_metrics


def generate_latex_table(aggregate_metrics, output_path=None, include_counts=False, use_display_names=False):
    """Generate a LaTeX table from aggregate metrics.
    
    Args:
        aggregate_metrics: Dictionary with aggregate metrics for each tokenizer
        output_path: Optional path to save the LaTeX file
        include_counts: Whether to include raw counts (TP, FP, FN) in the table
        use_display_names: Whether to use clean display names for models
        
    Returns:
        LaTeX table as a string
    """
    # Sort tokenizers by F1 score
    sorted_tokenizers = sorted(
        aggregate_metrics.keys(),
        key=lambda t: aggregate_metrics[t]["f1"],
        reverse=True
    )
    
    # Prepare LaTeX header
    if include_counts:
        latex = """\\begin{table}[htbp]
\\centering
\\caption{Aggregate Performance Across All Legal Datasets}
\\label{tab:aggregate-performance-with-counts}
\\begin{tabular}{lrrrrrrr}
\\toprule
Model & Precision & Recall & F1 & Time (ms/char) & TP & FP & FN \\\\
\\midrule
"""
    else:
        latex = """\\begin{table}[htbp]
\\centering
\\caption{Aggregate Performance Across All Legal Datasets}
\\label{tab:aggregate-performance}
\\begin{tabular}{lrrrr}
\\toprule
Model & Precision & Recall & F1 & Time (ms/char) \\\\
\\midrule
"""
    
    # Add rows for each tokenizer
    for tokenizer in sorted_tokenizers:
        metrics = aggregate_metrics[tokenizer]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        time_ms = metrics["time_per_char_ms"]
        
        # Use display name if requested
        display_name = MODEL_DISPLAY_NAMES.get(tokenizer, tokenizer) if use_display_names else tokenizer
        
        if include_counts:
            tp = metrics["raw_counts"]["true_positives"]
            fp = metrics["raw_counts"]["false_positives"]
            fn = metrics["raw_counts"]["false_negatives"]
            latex += f"{display_name} & {precision:.3f} & {recall:.3f} & {f1:.3f} & {time_ms:.3f} & {tp} & {fp} & {fn} \\\\\n"
        else:
            latex += f"{display_name} & {precision:.3f} & {recall:.3f} & {f1:.3f} & {time_ms:.3f} \\\\\n"
    
    # Close the LaTeX table
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Save if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"Aggregate metrics LaTeX table saved to {output_path}")
    
    return latex


def print_aggregate_metrics(aggregate_metrics, include_counts=False, use_display_names=False):
    """Print the aggregate metrics in a human-readable format."""
    print("\n=== Aggregate Metrics Across All Datasets ===")
    
    # Sort tokenizers by F1 score
    sorted_tokenizers = sorted(
        aggregate_metrics.keys(),
        key=lambda t: aggregate_metrics[t]["f1"],
        reverse=True
    )
    
    # Print header
    if include_counts:
        print(f"{'Model':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time (ms/char)':<15} {'TP':<10} {'FP':<10} {'FN':<10}")
        print("-" * 95)
    else:
        print(f"{'Model':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time (ms/char)':<15}")
        print("-" * 65)
    
    # Print each tokenizer
    for tokenizer in sorted_tokenizers:
        metrics = aggregate_metrics[tokenizer]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        time_ms = metrics["time_per_char_ms"]
        
        # Use display name if requested
        display_name = MODEL_DISPLAY_NAMES.get(tokenizer, tokenizer) if use_display_names else tokenizer
        
        if include_counts:
            tp = metrics["raw_counts"]["true_positives"]
            fp = metrics["raw_counts"]["false_positives"]
            fn = metrics["raw_counts"]["false_negatives"]
            print(f"{display_name:<20} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {time_ms:<15.3f} {tp:<10} {fp:<10} {fn:<10}")
        else:
            print(f"{display_name:<20} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {time_ms:<15.3f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate aggregate metrics table across all datasets")
    parser.add_argument("--input", type=str, default="results/paper_results_20250402_203406/evaluation_results.json",
                       help="Path to the evaluation results JSON file")
    parser.add_argument("--output", type=str, default="tables/aggregate_metrics_table.tex",
                       help="Path to save the LaTeX table")
    parser.add_argument("--with-counts", action="store_true",
                       help="Include raw counts (TP, FP, FN) in the output")
    parser.add_argument("--use-display-names", action="store_true",
                       help="Use clean display names for models")
    args = parser.parse_args()
    
    # Load the evaluation results
    try:
        with open(args.input, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in results file: {args.input}")
        sys.exit(1)
    
    # Calculate aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(results)
    
    # Print the metrics
    print_aggregate_metrics(aggregate_metrics, args.with_counts, args.use_display_names)
    
    # Generate and save LaTeX table - standard version
    generate_latex_table(aggregate_metrics, args.output, args.with_counts, False)
    
    # Generate version with display names
    display_names_output = args.output.replace(".tex", "_display_names.tex")
    generate_latex_table(aggregate_metrics, display_names_output, args.with_counts, True)
    print(f"Table with display names saved to {display_names_output}")
    
    # If counts were requested, also generate a version with counts
    if not args.with_counts:
        # Standard version with counts
        counts_output = args.output.replace(".tex", "_with_counts.tex")
        generate_latex_table(aggregate_metrics, counts_output, include_counts=True, use_display_names=False)
        print(f"Table with counts saved to {counts_output}")
        
        # Display names version with counts
        counts_display_output = args.output.replace(".tex", "_with_counts_display_names.tex")
        generate_latex_table(aggregate_metrics, counts_display_output, include_counts=True, use_display_names=True)
        print(f"Table with counts and display names saved to {counts_display_output}")
    
    print(f"\nAggregate metrics successfully processed from {args.input}")


if __name__ == "__main__":
    main()