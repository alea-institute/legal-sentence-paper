#!/usr/bin/env python3
"""
Regenerate HTML report from JSON results.

This script allows you to regenerate the HTML visualization report from an existing
evaluation_results.json file, without needing to rerun the entire evaluation.

Usage:
    python -m lsp.regenerate_html <input_json_file> <output_html_file>
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the HTML report generator functionality
from lsp.visualization_html import process_evaluation_results


def generate_custom_html_report(results, output_path, max_examples_per_dataset=10):
    """Custom implementation of HTML report generation with enhanced chart finding.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save the HTML report
        max_examples_per_dataset: Maximum number of examples to include per dataset
    """
    from jinja2 import Environment, FileSystemLoader
    
    # Process the evaluation results
    processed_data = process_evaluation_results(results, max_examples_per_dataset)
    
    # Find all potential chart images with more expansive searching
    input_dir = os.path.dirname(os.path.abspath(output_path))
    
    # Define potential directories to check for charts
    chart_dirs = [
        os.path.join(input_dir, "final_publication_charts"),   # Look for final charts first
        os.path.join(input_dir, "updated_publication_charts"),  # Then updated charts
        os.path.join(input_dir, "charts"),
        os.path.join(input_dir, "publication_charts"),
        os.path.join(input_dir, "publication_charts_final"),
        # Also try one level up
        os.path.join(os.path.dirname(input_dir), "charts"),
        os.path.join(os.path.dirname(input_dir), "publication_charts")
    ]
    
    # Function to encode image as base64
    def encode_image(file_path):
        with open(file_path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    
    # Dictionary to store all chart paths
    chart_paths = {}
    
    # Print diagnostic info
    print("\nSearching for charts in the following directories:")
    for chart_dir in chart_dirs:
        if os.path.exists(chart_dir):
            print(f"  - {chart_dir} (exists)")
        else:
            print(f"  - {chart_dir} (not found)")
    
    # Search all directories for charts
    for chart_dir in chart_dirs:
        if not os.path.exists(chart_dir):
            continue
        
        print(f"\nFound {len(os.listdir(chart_dir))} files in {chart_dir}")
        
        # Check for standard comparison charts
        comparison_dir = os.path.join(chart_dir, "comparison")
        if os.path.exists(comparison_dir):
            for metric in ["f1", "precision", "recall", "time-per-char-seconds", "time-per-sentence-seconds"]:
                chart_path = os.path.join(comparison_dir, f"{metric}.png")
                if os.path.exists(chart_path):
                    print(f"  Found comparison chart: {chart_path}")
                    chart_paths[metric] = encode_image(chart_path)
        
        # Look for charts in the root directory that match common patterns
        for filename in os.listdir(chart_dir):
            file_path = os.path.join(chart_dir, filename)
            
            # Skip if not a file or not a PNG
            if not os.path.isfile(file_path) or not filename.endswith('.png'):
                continue
                
            print(f"  Examining chart: {filename}")
            
            # Handle various chart types and naming patterns
            
            # Comparison charts
            for metric in ["f1", "precision", "recall", "time_per_char_seconds", "time-per-char-seconds"]:
                if f"{metric}.png" == filename or f"{metric}_heatmap.png" == filename:
                    chart_key = metric.replace('_', '-')
                    if chart_key not in chart_paths:
                        chart_paths[chart_key] = encode_image(file_path)
                        print(f"    Added as {chart_key}")
            
            # Tradeoff charts
            if "f1_vs_throughput.png" in filename or "f1_time_tradeoff.png" in filename:
                chart_paths["tradeoff_f1_vs_throughput"] = encode_image(file_path)
                print(f"    Added as tradeoff_f1_vs_throughput")
            elif "precision_vs_throughput.png" in filename or "precision_vs_time.png" in filename:
                chart_paths["tradeoff_precision_vs_throughput"] = encode_image(file_path)
                print(f"    Added as tradeoff_precision_vs_throughput")
            elif "recall_vs_throughput.png" in filename or "recall_vs_time.png" in filename:
                chart_paths["tradeoff_recall_vs_throughput"] = encode_image(file_path)
                print(f"    Added as tradeoff_recall_vs_throughput")
            elif "precision_recall.png" in filename or "precision_recall_tradeoff.png" in filename:
                chart_paths["tradeoff_precision_recall"] = encode_image(file_path)
                print(f"    Added as tradeoff_precision_recall")
            
            # Dataset comparison
            if "dataset_comparison.png" in filename:
                chart_paths["dataset_comparison"] = encode_image(file_path)
                print(f"    Added as dataset_comparison")
            
            # Weighted metrics
            if "weighted_metrics.png" in filename:
                chart_paths["weighted_metrics"] = encode_image(file_path)
                print(f"    Added as weighted_metrics")
                
            # Heatmaps
            for metric in ["f1", "precision", "recall", "time_per_char_seconds"]:
                if f"{metric}_heatmap.png" == filename:
                    chart_paths[f"{metric}_heatmap"] = encode_image(file_path)
                    print(f"    Added as {metric}_heatmap")
        
        # Check tradeoffs directory
        tradeoffs_dir = os.path.join(chart_dir, "tradeoffs")
        if os.path.exists(tradeoffs_dir):
            for metric_pair in ["f1_vs_throughput", "precision_vs_throughput", "recall_vs_throughput", "precision_recall"]:
                chart_path = os.path.join(tradeoffs_dir, f"{metric_pair}.png")
                if os.path.exists(chart_path):
                    print(f"  Found tradeoff chart: {chart_path}")
                    chart_paths[f"tradeoff_{metric_pair}"] = encode_image(chart_path)
    
    # Add chart paths to processed data
    processed_data["chart_paths"] = chart_paths
    print(f"\nFound {len(chart_paths)} charts to include in the report")
    
    # Load template
    template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lsp', 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('evaluation_results.html')
    
    # Render template
    html_content = template.render(**processed_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate HTML report from evaluation results JSON")
    parser.add_argument("input_json", help="Path to evaluation_results.json file")
    parser.add_argument("output_html", help="Path to save the HTML report")
    parser.add_argument("--max-examples", type=int, default=10, 
                      help="Maximum number of examples to show per dataset")
    parser.add_argument("--verbose", "-v", action="store_true", 
                      help="Print verbose information about chart discovery")
    args = parser.parse_args()
    
    # Ensure the input file exists
    if not os.path.exists(args.input_json):
        print(f"Error: Input file '{args.input_json}' not found.")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    
    try:
        # Load the evaluation results
        print(f"Loading evaluation results from {args.input_json}")
        with open(args.input_json, 'r') as f:
            results = json.load(f)
        
        # Generate the HTML report with enhanced chart discovery
        print(f"Generating HTML report with up to {args.max_examples} examples per dataset")
        generate_custom_html_report(results, args.output_html, max_examples_per_dataset=args.max_examples)
        
        print(f"HTML report successfully generated at {args.output_html}")
        
    except json.JSONDecodeError:
        print(f"Error: The file '{args.input_json}' is not valid JSON.")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating HTML report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()