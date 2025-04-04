#!/usr/bin/env python3
"""
Specialized script to create a clean precision-recall plot for the paper.

This creates a simple, clean precision-recall plot with:
- Full [0,1] range for both precision and recall
- F1 isolines
- Colors matching our other charts
- Legend outside the chart area
- Neat, professional appearance
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import color map from visualization module
try:
    from lsp.visualization import get_tokenizer_color_map
except ImportError:
    print("Error: Unable to import from lsp.visualization. Make sure the package is installed.")
    sys.exit(1)

# Define color scheme for consistency
TOKENIZER_COLORS = {
    # Our methods in blues
    "nupunkt": "#4E79A7",  # Muted blue
    "charboundary_small": "#86BCB6",  # Light teal blue
    "charboundary_medium": "#5D9AA7",  # Medium teal blue 
    "charboundary_large": "#2A7B9B",  # Dark teal blue
    
    # Competitors in other colors
    "nltk_punkt": "#B07AA1",  # Dusty plum
    "pysbd": "#E15759",  # Muted terracotta
    "spacy_sm": "#8CD17D",  # Lighter muted green
    "spacy_lg": "#2E7D39",  # Darker muted green
}

# Set plot style for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'Computer Modern Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})


def create_precision_recall_plot(results, output_path, width=6.0, height=6.0):
    """Create a clean precision-recall plot following the specified requirements.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save the output files
        width: Figure width in inches
        height: Figure height in inches
    """
    # Extract data
    tokenizers = results['tokenizers']
    datasets = results['datasets']
    
    # Create figure with square dimensions and adjusted padding
    # With legend at bottom, need some bottom padding but keep the plot square
    fig, ax = plt.subplots(figsize=(width, height))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.22, top=0.95)  # Increased bottom padding for x-label
    
    # Get color map for tokenizers
    color_map = TOKENIZER_COLORS
    
    # Calculate metrics by combining raw counts across all datasets
    data = []
    for tokenizer_name in tokenizers:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for dataset_name in datasets:
            try:
                dataset_result = results["results"][tokenizer_name][dataset_name]
                if "summary" in dataset_result:
                    summary = dataset_result["summary"]
                    
                    # Check for both naming conventions in summary
                    if "total_true_positives" in summary:
                        total_tp += summary["total_true_positives"]
                        total_fp += summary["total_false_positives"] 
                        total_fn += summary["total_false_negatives"]
                    elif "true_positives" in summary:
                        total_tp += summary["true_positives"]
                        total_fp += summary["false_positives"]
                        total_fn += summary["false_negatives"]
            except Exception:
                pass
        
        # Calculate precision, recall, F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Add to data collection
        data.append({
            "tokenizer": tokenizer_name,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
    
    # Draw F1 isolines
    f1_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    recall_points = np.linspace(0.01, 0.99, 200)
    
    for f1 in f1_levels:
        precision_points = []
        valid_recalls = []
        
        for r in recall_points:
            # Formula: precision = (F1 * recall) / (2 * recall - F1)
            if 2 * r - f1 > 0:
                p = (f1 * r) / (2 * r - f1)
                if 0 <= p <= 1:
                    precision_points.append(p)
                    valid_recalls.append(r)
        
        if precision_points:
            # Plot isoline
            ax.plot(valid_recalls, precision_points, '--', color='#CCCCCC', linewidth=0.8, alpha=0.6)
            
            # Label the line (but only for select F1 values to avoid clutter)
            if f1 in [0.3, 0.5, 0.7, 0.9]:
                midpoint_idx = len(valid_recalls) // 2
                if midpoint_idx < len(valid_recalls):
                    ax.annotate(
                        f"F1={f1:.1f}",
                        (valid_recalls[midpoint_idx], precision_points[midpoint_idx]),
                        fontsize=7,
                        color='#999999',
                        ha='center',
                        va='center',
                        fontstyle='italic'
                    )
    
    # Reorder data for legend grouping, putting nupunkt with charboundary models first
    ordered_tokens = []
    our_models = [t for t in tokenizers if 'nupunkt' in t or 'charboundary' in t]
    other_models = [t for t in tokenizers if t not in our_models]
    ordered_tokens.extend(our_models)
    ordered_tokens.extend(other_models)
    
    # Sort data according to our preferred order
    ordered_data = []
    for token in ordered_tokens:
        for entry in data:
            if entry["tokenizer"] == token:
                ordered_data.append(entry)
                break
    
    # Define marker styles for different model families
    marker_styles = {
        'nupunkt': 'o',
        'charboundary_small': '^',  # triangle up
        'charboundary_medium': 's',  # square
        'charboundary_large': 'D',   # diamond
        'nltk_punkt': 'o',
        'spacy_sm': '^',   # triangle up
        'spacy_lg': 's',   # square
        'pysbd': 'o'
    }
    
    # Plot data points
    for entry in ordered_data:
        tokenizer = entry["tokenizer"]
        precision = entry["precision"]
        recall = entry["recall"]
        f1 = entry["f1"]
        
        # Get color for this tokenizer
        color = color_map.get(tokenizer, "#333333")
        
        # Define marker style - use different markers by model type
        marker = marker_styles.get(tokenizer, 'o')
        
        # Generate label with F1 score on second line
        label = f"{tokenizer}\nF1={f1:.2f}"
        
        # Plot the point with half the size marker
        ax.scatter(
            recall, 
            precision, 
            c=color,
            marker=marker,
            s=60,  # Half the size of previous marker
            alpha=0.7,  # More transparency to see overlap
            edgecolors='white',
            linewidth=0.5,
            label=label,
            zorder=10  # Ensure points are above grid lines
        )
    
    # Set axis limits to [0.5,1.0] range with small padding
    ax.set_xlim(0.49, 1.01)
    ax.set_ylim(0.49, 1.01)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.2)
    
    # Set regular tick marks at 0.1 intervals, but customized for the [0.5,1.0] range
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    
    # Set specific tick positions appropriate for the [0.5,1.0] range
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Style the plot
    ax.set_xlabel('Recall', fontsize=11, fontweight='medium')
    ax.set_ylabel('Precision', fontsize=11, fontweight='medium')
    ax.set_title('Precision-Recall Performance', fontsize=12, fontweight='medium', pad=10)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create the legend OUTSIDE the plot
    # This places it horizontally at the bottom of the plot
    legend = ax.legend(
        bbox_to_anchor=(0.5, -0.15),  # Move legend further down 
        loc='upper center',
        ncol=2,  # 2 columns puts more related models in the same column
        frameon=True,
        fancybox=False,
        edgecolor='#CCCCCC',
        fontsize=7,  # Even smaller font size
        title="Models",
        title_fontsize=8,  # Smaller title font size
        labelspacing=0.3,  # Tighter spacing between legend items
        columnspacing=2.5,  # Slightly less space between columns
        handletextpad=0.3,  # Less space between marker and text
        borderpad=0.5  # Slightly less padding inside the legend border
    )
    
    # We've already set subplot adjustments at figure creation, so no need to call tight_layout
    # Just ensure we have space for legend (right=0.65 set at figure creation)
    
    # Save figure in both formats
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    # Close the figure
    plt.close(fig)
    
    print(f"Created precision-recall plot: {output_path}")
    print(f"PDF version saved to: {pdf_path}")
    
    return output_path


def main():
    """Main function to parse arguments and create the plot."""
    parser = argparse.ArgumentParser(description="Generate a clean precision-recall plot")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to the evaluation results JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output PNG file")
    parser.add_argument("--width", type=float, default=5.0,
                        help="Width of the figure in inches")
    parser.add_argument("--height", type=float, default=4.0,
                        help="Height of the figure in inches")
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
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create the plot
    create_precision_recall_plot(results, args.output, args.width, args.height)


if __name__ == "__main__":
    main()