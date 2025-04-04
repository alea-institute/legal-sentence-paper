#!/usr/bin/env python3
"""
Generate custom tradeoff charts from evaluation results.

This script loads evaluation results from a JSON file and generates various
tradeoff charts to visualize the performance/throughput tradeoffs between
different tokenizers.

Usage:
    python -m lsp.examples.generate_tradeoff_charts [--input RESULTS_JSON] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import visualization functions
try:
    import matplotlib.pyplot as plt
    from lsp.visualization import (
        plot_custom_tradeoff_chart,
        create_dataset_specific_chart,
        plot_precision_recall_tradeoff
    )
except ImportError:
    print("Error: This script requires matplotlib and numpy to be installed.")
    print("Run: pip install matplotlib numpy")
    sys.exit(1)


def main():
    """Main function to generate tradeoff charts."""
    parser = argparse.ArgumentParser(description="Generate custom tradeoff charts")
    parser.add_argument("--input", type=str, default="results/evaluation_results.json",
                        help="Path to the evaluation results JSON file")
    parser.add_argument("--output", type=str, default="results/charts/tradeoffs",
                        help="Directory to save output charts")
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
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get the list of datasets and tokenizers
    datasets = results.get("datasets", [])
    tokenizers = results.get("tokenizers", [])
    
    if not datasets or not tokenizers:
        print("Error: No datasets or tokenizers found in results file.")
        sys.exit(1)
    
    print(f"Found {len(tokenizers)} tokenizers and {len(datasets)} datasets.")
    
    # 1. Generate precision vs time tradeoff chart (aggregated across datasets)
    print("Generating precision vs time tradeoff chart...")
    precision_vs_time_path = os.path.join(args.output, "precision_vs_time.png")
    fig, ax = plot_custom_tradeoff_chart(
        results,
        x_metric="time_per_char_seconds",
        y_metric="precision",
        output_path=precision_vs_time_path,
        aggregate_method="weighted",
        invert_x=True,
        title="Precision vs Time per Character Tradeoff"
    )
    plt.close(fig)
    
    # 2. Generate F1 vs time tradeoff chart (aggregated across datasets)
    print("Generating F1 vs time tradeoff chart...")
    f1_vs_time_path = os.path.join(args.output, "f1_vs_time.png")
    fig, ax = plot_custom_tradeoff_chart(
        results,
        x_metric="time_per_char_seconds",
        y_metric="f1",
        output_path=f1_vs_time_path,
        aggregate_method="weighted",
        invert_x=True,
        title="F1 Score vs Time per Character Tradeoff"
    )
    plt.close(fig)
    
    # 3. Generate recall vs time tradeoff chart (aggregated across datasets)
    print("Generating recall vs time tradeoff chart...")
    recall_vs_time_path = os.path.join(args.output, "recall_vs_time.png")
    fig, ax = plot_custom_tradeoff_chart(
        results,
        x_metric="time_per_char_seconds",
        y_metric="recall",
        output_path=recall_vs_time_path,
        aggregate_method="weighted",
        invert_x=True,
        title="Recall vs Time per Character Tradeoff"
    )
    plt.close(fig)
    
    # 4. Generate the classic precision-recall tradeoff chart
    print("Generating precision-recall tradeoff chart...")
    precision_recall_path = os.path.join(args.output, "precision_recall.png")
    fig, ax = plot_precision_recall_tradeoff(
        results,
        output_path=precision_recall_path
    )
    plt.close(fig)
    
    # 5. Generate per-dataset charts
    datasets_dir = os.path.join(args.output, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    print("Generating per-dataset tradeoff charts...")
    for dataset_name in datasets:
        # Create F1 vs time chart for this dataset
        dataset_output_path = os.path.join(datasets_dir, f"{dataset_name}_f1_vs_time.png")
        fig, ax = create_dataset_specific_chart(
            results,
            dataset_name=dataset_name,
            x_metric="time_per_char_seconds",
            y_metric="f1",
            output_path=dataset_output_path,
            title=f"F1 vs Processing Time for {dataset_name}",
            invert_x=True
        )
        plt.close(fig)
    
    print(f"All charts have been saved to: {args.output}")
    print("To examine dataset-specific charts, check the 'datasets' subdirectory.")
    print("\nExample usage of custom tradeoff charts in your code:")
    print("""
# Import functions
from lsp.visualization import plot_custom_tradeoff_chart, create_dataset_specific_chart

# Load your results data
with open("results.json", "r") as f:
    results = json.load(f)

# Create a custom chart comparing any two metrics
fig, ax = plot_custom_tradeoff_chart(
    results,
    x_metric="time_per_char_seconds",  # X-axis metric
    y_metric="precision",              # Y-axis metric
    aggregate_method="weighted",       # How to aggregate data points: "weighted", "mean", or "none"
    filter_datasets=["dataset1"],      # Optional: filter to specific datasets
    filter_tokenizers=["tokenizer1"],  # Optional: filter to specific tokenizers
    invert_x=True,                     # Invert X-axis (useful for time metrics where lower is better)
    title="My Custom Chart"            # Custom title
)

# Save the chart
plt.savefig("my_chart.png", dpi=300, bbox_inches='tight')
plt.close(fig)
""")


if __name__ == "__main__":
    main()