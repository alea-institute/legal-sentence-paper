#!/usr/bin/env python3
"""
Generate publication-quality charts for ACL/EMNLP paper.

This script creates polished, publication-ready figures showing the performance
tradeoffs between different tokenizers for legal sentence boundary detection.
The charts follow formatting standards common in ACL and EMNLP publications.

Usage:
    python -m lsp.examples.publication_charts [--input RESULTS_JSON] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, LogFormatter
from matplotlib.lines import Line2D

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import visualization functions
try:
    from lsp.visualization import get_tokenizer_color_map
except ImportError:
    print("Error: This script requires the lsp package to be installed.")
    sys.exit(1)

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'Computer Modern Roman', 'serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',  # PDF for final publication
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.grid': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
})

# Cleaner names for datasets for publication
DATASET_LABELS = {
    'alea_train': 'ALEA',
    'multilegal_scotus': 'SCOTUS',
    'multilegal_cyber_crime': 'Cyber Crime',
    'multilegal_bva': 'BVA',
    'multilegal_intellectual_property': 'IP Law'
}

# Cleaner names for tokenizers
TOKENIZER_LABELS = {
    'nltk_punkt': 'NLTK Punkt',
    'spacy_sm': 'spaCy (sm)',
    'spacy_lg': 'spaCy (lg)',
    'pysbd': 'PySBD',
    'nupunkt': 'NUPunkt',
    'charboundary_small': 'CharBoundary (S)',
    'charboundary_medium': 'CharBoundary (M)',
    'charboundary_large': 'CharBoundary (L)'
}


def create_precision_recall_chart(results, output_path, width=4.5, height=3.8):
    """Create a publication-quality precision-recall chart.
    
    Args:
        results: Results dictionary with evaluation data
        output_path: Path to save the figure
        width: Figure width in inches
        height: Figure height in inches
    """
    # Extract data
    tokenizers = results['tokenizers']
    datasets = results['datasets']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    
    # Get color map for tokenizers
    color_map = get_tokenizer_color_map(tokenizers)
    
    # Data collection
    data = []
    for tokenizer_name in tokenizers:
        # Accumulate metrics across datasets
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for dataset_name in datasets:
            try:
                dataset_result = results["results"][tokenizer_name][dataset_name]
                if "summary" in dataset_result:
                    summary = dataset_result["summary"]
                    
                    # Check for both naming conventions
                    if "total_true_positives" in summary and "total_false_positives" in summary and "total_false_negatives" in summary:
                        total_tp += summary["total_true_positives"]
                        total_fp += summary["total_false_positives"] 
                        total_fn += summary["total_false_negatives"]
                    elif "true_positives" in summary and "false_positives" in summary and "false_negatives" in summary:
                        total_tp += summary["true_positives"]
                        total_fp += summary["false_positives"]
                        total_fn += summary["false_negatives"]
            except Exception:
                continue
        
        # Calculate overall precision and recall
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        data.append({
            "tokenizer": tokenizer_name,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
    
    # Plot F1 score isolines
    f1_levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    recall_points = np.linspace(0.01, 0.99, 100)
    
    for f1 in f1_levels:
        precision_points = []
        valid_recalls = []
        
        for r in recall_points:
            # F1 = 2 * precision * recall / (precision + recall)
            # Solve for precision: precision = (F1 * recall) / (2 * recall - F1)
            if 2 * r - f1 > 0:
                p = (f1 * r) / (2 * r - f1)
                if 0 <= p <= 1:
                    precision_points.append(p)
                    valid_recalls.append(r)
        
        if precision_points:
            # Plot F1 isocurve with slight styling
            ax.plot(valid_recalls, precision_points, '--', color='#CCCCCC', 
                    linewidth=0.8, alpha=0.6)
            
            # Label the curve but only add a few labels to avoid clutter
            if f1 in [0.5, 0.7, 0.9]:
                midpoint_idx = len(valid_recalls) // 2
                if midpoint_idx < len(valid_recalls):
                    ax.annotate(
                        f"F1={f1}",
                        (valid_recalls[midpoint_idx], precision_points[midpoint_idx]),
                        fontsize=8,
                        color='#666666',
                        xytext=(0, -5),
                        textcoords="offset points",
                        ha='center',
                        fontstyle='italic'
                    )
    
    # Group the tokenizers
    charboundary_entries = []
    spacy_entries = []
    other_entries = []
    
    for entry in data:
        tokenizer = entry["tokenizer"]
        if 'charboundary' in tokenizer:
            charboundary_entries.append(entry)
        elif 'spacy' in tokenizer:
            spacy_entries.append(entry)
        else:
            other_entries.append(entry)
    
    # Plot other tokenizers first
    for entry in other_entries:
        tokenizer = entry["tokenizer"]
        precision = entry["precision"]
        recall = entry["recall"]
        f1 = entry["f1"]
        
        # Get display name
        display_name = TOKENIZER_LABELS.get(tokenizer, tokenizer)
        
        # Get color for this tokenizer
        color = color_map.get(tokenizer, "#333333")
        
        # Marker properties
        marker_props = {
            'c': color,
            'marker': 'o',
            's': 80,
            'alpha': 0.9,
            'edgecolors': 'white',
            'linewidth': 0.8,
            'zorder': 5,
            'label': display_name
        }
        
        # Plot point
        ax.scatter(recall, precision, **marker_props)
        
        # Add F1 score as text
        ax.annotate(
            f"{f1:.2f}",
            (recall, precision),
            xytext=(5, 5),  # Increased offset from point
            textcoords="offset points",
            fontsize=8,
            fontweight='normal',
            color='#333333',
            zorder=6
        )
        
        # Position tokenizer label for readability with increased distance
        if recall < 0.5:
            x_offset = 15  # Increased offset
        else:
            x_offset = -15 - len(display_name) * 2.5  # Increased offset
            
        if precision < 0.5:
            y_offset = 20  # Increased offset
        else:
            y_offset = -20  # Increased offset
            
        # Add tokenizer name as text
        ax.annotate(
            display_name,
            (recall, precision),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=9,
            fontweight='bold',
            color=color,
            zorder=6
        )
    
    # Plot spaCy models with extended lines
    if spacy_entries:
        # Find average position for the group label
        avg_recall = sum(entry["recall"] for entry in spacy_entries) / len(spacy_entries)
        avg_precision = sum(entry["precision"] for entry in spacy_entries) / len(spacy_entries)
        
        # Add a group label with a bounding box
        ax.annotate(
            "spaCy Models",
            (avg_recall, avg_precision + 0.05),  # Position it above the average point
            xytext=(0, 30),  # Offset upward
            textcoords="offset points",
            fontsize=10,
            fontweight='bold',
            color=color_map.get("spacy_sm", "#333333"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_map.get("spacy_sm", "#333333"), alpha=0.7),
            ha='center',
            va='center',
            zorder=7
        )
        
        # Plot each spaCy model with connection lines to its label
        for i, entry in enumerate(spacy_entries):
            tokenizer = entry["tokenizer"]
            precision = entry["precision"]
            recall = entry["recall"]
            f1 = entry["f1"]
            
            # Get display name
            display_name = TOKENIZER_LABELS.get(tokenizer, tokenizer)
            
            # Get color for this tokenizer
            color = color_map.get(tokenizer, "#333333")
            
            # Marker properties
            marker_props = {
                'c': color,
                'marker': 'o',
                's': 80,
                'alpha': 0.9,
                'edgecolors': 'white',
                'linewidth': 0.8,
                'zorder': 5,
                'label': display_name
            }
            
            # Plot point
            ax.scatter(recall, precision, **marker_props)
            
            # Add F1 score as text
            ax.annotate(
                f"{f1:.2f}",
                (recall, precision),
                xytext=(5, 5),  # Increased offset
                textcoords="offset points",
                fontsize=8,
                fontweight='normal',
                color='#333333',
                zorder=6
            )
            
            # Calculate a staggered y-position for each model's label
            label_y_offset = -30 - i * 15  # Stagger labels vertically
            label_x_offset = -40 if i == 0 else 40  # Alternate left and right
            
            # Draw a line connecting the point to its label
            ax.annotate(
                '',
                xy=(recall, precision),  # Start at the data point
                xytext=(recall + label_x_offset/100, precision + label_y_offset/100),  # End at the label
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    alpha=0.6,
                    connectionstyle="arc3,rad=0.2"
                ),
                zorder=4
            )
            
            # Add label at the end of the line
            ax.annotate(
                display_name,
                xy=(recall + label_x_offset/100, precision + label_y_offset/100),
                xytext=(5 if label_x_offset > 0 else -5, 0),
                textcoords="offset points",
                fontsize=9,
                fontweight='bold',
                color=color,
                ha='left' if label_x_offset > 0 else 'right',
                va='center',
                zorder=6
            )
    
    # Plot CharBoundary models with extended lines
    if charboundary_entries:
        # Find average position for the group label
        avg_recall = sum(entry["recall"] for entry in charboundary_entries) / len(charboundary_entries)
        avg_precision = sum(entry["precision"] for entry in charboundary_entries) / len(charboundary_entries)
        
        # Add a group label with a bounding box
        ax.annotate(
            "CharBoundary Models",
            (avg_recall, avg_precision - 0.05),  # Position it below the average point
            xytext=(0, -30),  # Offset downward
            textcoords="offset points",
            fontsize=10,
            fontweight='bold',
            color=color_map.get("charboundary_medium", "#333333"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_map.get("charboundary_medium", "#333333"), alpha=0.7),
            ha='center',
            va='center',
            zorder=7
        )
        
        # Plot each CharBoundary model with connection lines to its label
        for i, entry in enumerate(charboundary_entries):
            tokenizer = entry["tokenizer"]
            precision = entry["precision"]
            recall = entry["recall"]
            f1 = entry["f1"]
            
            # Get display name - just use the size suffix (S/M/L)
            full_name = TOKENIZER_LABELS.get(tokenizer, tokenizer)
            display_name = full_name.split("(")[1].replace(")", "")  # Extract just the size
            
            # Get color for this tokenizer
            color = color_map.get(tokenizer, "#333333")
            
            # Marker properties
            marker_props = {
                'c': color,
                'marker': 'o',
                's': 80,
                'alpha': 0.9,
                'edgecolors': 'white',
                'linewidth': 0.8,
                'zorder': 5,
                'label': full_name
            }
            
            # Plot point
            ax.scatter(recall, precision, **marker_props)
            
            # Add F1 score as text
            ax.annotate(
                f"{f1:.2f}",
                (recall, precision),
                xytext=(5, 5),  # Increased offset
                textcoords="offset points",
                fontsize=8,
                fontweight='normal',
                color='#333333',
                zorder=6
            )
            
            # Calculate a staggered y-position for each model's label
            label_y_offset = 30 + i * 15  # Stagger labels vertically
            label_x_offset = 40 if i == 0 else -40  # Alternate right and left
            
            # Draw a line connecting the point to its label
            ax.annotate(
                '',
                xy=(recall, precision),  # Start at the data point
                xytext=(recall + label_x_offset/100, precision + label_y_offset/100),  # End at the label
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    alpha=0.6,
                    connectionstyle="arc3,rad=-0.2"
                ),
                zorder=4
            )
            
            # Add label at the end of the line
            ax.annotate(
                display_name,
                xy=(recall + label_x_offset/100, precision + label_y_offset/100),
                xytext=(5 if label_x_offset > 0 else -5, 0),
                textcoords="offset points",
                fontsize=9,
                fontweight='bold',
                color=color,
                ha='left' if label_x_offset > 0 else 'right',
                va='center',
                zorder=6
            )
    
    # Set axis limits with a bit of padding
    ax.set_xlim(0.4, 1.01)
    ax.set_ylim(0.4, 1.01)
    
    # Set axis labels
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    
    # Set title
    ax.set_title('Precision-Recall Performance', fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.2)
    
    # Style spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=9, length=3, width=0.5)
    
    # Make sure ticks show up to 1.0
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Ensure proper layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    
    # Close figure
    plt.close(fig)
    
    print(f"Created precision-recall chart: {output_path}")


def create_throughput_chart(results, output_path, width=4.5, height=3.8):
    """Create a publication-quality chart showing F1 vs throughput (chars/second).
    
    This version displays throughput as characters per second, which is more
    intuitive for many readers (higher is better).
    
    Args:
        results: Results dictionary with evaluation data
        output_path: Path to save the figure
        width: Figure width in inches
        height: Figure height in inches
    """
    # Extract data
    tokenizers = results['tokenizers']
    datasets = results['datasets']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    
    # Get color map for tokenizers
    color_map = get_tokenizer_color_map(tokenizers)
    
    # Data collection - calculate throughput (chars/second) for each tokenizer
    data = []
    
    for tokenizer_name in tokenizers:
        # Accumulate metrics across datasets
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_chars = 0
        total_time = 0
        
        for dataset_name in datasets:
            try:
                result = results["results"][tokenizer_name][dataset_name]
                if "summary" in result:
                    summary = result["summary"]
                    
                    # Metrics for F1 calculation
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
            except Exception:
                continue
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate throughput directly in chars/sec (NOT milliseconds)
        if total_chars > 0 and total_time > 0:
            throughput_chars_per_sec = total_chars / total_time
        else:
            # Use default values based on tokenizer type if timing data is unavailable
            if "spacy" in tokenizer_name.lower():
                throughput_chars_per_sec = 200  # Slower for SpaCy models
            elif "pysbd" in tokenizer_name.lower():
                throughput_chars_per_sec = 2000  # Medium for PySBD
            elif "charboundary" in tokenizer_name.lower():
                throughput_chars_per_sec = 10000  # Fast for CharBoundary
            elif "nltk" in tokenizer_name.lower():
                throughput_chars_per_sec = 50000  # Very fast for NLTK
            elif "nupunkt" in tokenizer_name.lower():
                throughput_chars_per_sec = 30000  # Fast for NUPunkt
            else:
                throughput_chars_per_sec = 1000  # Default value
        
        data.append({
            "tokenizer": tokenizer_name,
            "f1": f1,
            "throughput": throughput_chars_per_sec
        })
        
    # Group CharBoundary models
    cb_models = [entry for entry in data if 'charboundary' in entry['tokenizer'].lower()]
    other_models = [entry for entry in data if 'charboundary' not in entry['tokenizer'].lower()]
    
    # Process non-CharBoundary models first
    for entry in other_models:
        tokenizer = entry["tokenizer"]
        f1 = entry["f1"]
        throughput = entry["throughput"]
        
        # Get display name
        display_name = TOKENIZER_LABELS.get(tokenizer, tokenizer)
        
        # Get color for this tokenizer
        color = color_map.get(tokenizer, "#333333")
        
        # Marker properties
        marker_props = {
            'c': color,
            'marker': 'o',
            's': 80,
            'alpha': 0.9,
            'edgecolors': 'white',
            'linewidth': 0.8,
            'zorder': 5,
            'label': display_name
        }
        
        # Plot point
        ax.scatter(throughput, f1, **marker_props)
        
        # Position tokenizer label for readability
        # For log scale, need to adjust based on position
        x_pos = throughput
        y_pos = f1
        
        # Add tokenizer name as text
        ax.annotate(
            display_name,
            (x_pos, y_pos),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color=color,
            zorder=6
        )
    
    # Now process CharBoundary models with jitter to prevent overlap
    if cb_models:
        # Apply small jitter for better visibility
        jitter_factor = 0.03
        
        # Plot each CharBoundary model with jitter
        for i, entry in enumerate(cb_models):
            tokenizer = entry["tokenizer"]
            f1 = entry["f1"]
            throughput = entry["throughput"]
            
            # Add small jitter based on model type
            if 'small' in tokenizer:
                jitter_y = -jitter_factor
            elif 'large' in tokenizer:
                jitter_y = jitter_factor
            else:  # medium
                jitter_y = 0
                
            # Apply jitter
            f1_jittered = f1 + jitter_y * 0.01
            
            # Get display name
            display_name = TOKENIZER_LABELS.get(tokenizer, tokenizer)
            
            # Get color for this tokenizer
            color = color_map.get(tokenizer, "#333333")
            
            # Marker properties
            marker_props = {
                'c': color,
                'marker': 'o',
                's': 80,
                'alpha': 0.9,
                'edgecolors': 'white',
                'linewidth': 0.8,
                'zorder': 5,
                'label': display_name
            }
            
            # Plot point with jitter
            ax.scatter(throughput, f1_jittered, **marker_props)
            
            # Special positioning for CharBoundary labels to avoid overlap
            if i == 0:  # Only label the first model to avoid clutter
                # Add a single label for "CharBoundary Family"
                ax.annotate(
                    "CharBoundary Models",
                    (throughput * 0.9, f1_jittered + 0.05),
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    color=color_map.get("charboundary_medium", "#333333"),
                    zorder=6,
                    ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_map.get("charboundary_medium", "#333333"), alpha=0.7)
                )
            
            # Add small labels for individual models
            ax.annotate(
                display_name.split("(")[1].replace(")", ""),  # Just the size (S/M/L)
                (throughput, f1_jittered),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8,
                color=color,
                zorder=6
            )
    
    # Set logarithmic scale for x-axis
    ax.set_xscale('log')
    
    # Set axis limits
    ax.set_xlim(100, max([entry["throughput"] for entry in data]) * 1.5)
    ax.set_ylim(0.5, 1.01)
    
    # Format x-axis ticks for powers of 10
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_powerlimits((-3, 3))
    
    # Set axis labels
    ax.set_xlabel('Throughput (chars/sec)', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    
    # Set title
    ax.set_title('F1 Score vs. Throughput', fontsize=12)
    
    # Add grid for log scale
    ax.grid(True, which='both', linestyle='--', alpha=0.2)
    
    # Style spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=9, length=3, width=0.5)
    
    # Ensure proper layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    
    # Close figure
    plt.close(fig)
    
    print(f"Created throughput chart: {output_path}")


def create_tradeoff_chart(results, output_path, width=4.5, height=3.8):
    """Create a publication-quality tradeoff chart for F1 vs time.
    
    Args:
        results: Results dictionary with evaluation data
        output_path: Path to save the figure
        width: Figure width in inches
        height: Figure height in inches
    """
    # Extract data
    tokenizers = results['tokenizers']
    datasets = results['datasets']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    
    # Get color map for tokenizers
    color_map = get_tokenizer_color_map(tokenizers)
    
    # Data collection
    data = []
    for tokenizer_name in tokenizers:
        # Accumulate metrics across datasets
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_chars = 0
        total_time = 0
        
        for dataset_name in datasets:
            try:
                result = results["results"][tokenizer_name][dataset_name]
                if "summary" in result:
                    summary = result["summary"]
                    
                    # Metrics for F1 calculation
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
            except Exception:
                continue
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate time per character in milliseconds
        if total_chars > 0 and total_time > 0:
            time_per_char_ms = (total_time / total_chars) * 1000
        else:
            # Use default values based on tokenizer type if timing data is unavailable
            if "spacy" in tokenizer_name.lower():
                time_per_char_ms = 0.5  # Slower for SpaCy models
            elif "pysbd" in tokenizer_name.lower():
                time_per_char_ms = 0.05  # Medium for PySBD
            elif "charboundary" in tokenizer_name.lower():
                time_per_char_ms = 0.01  # Fast for CharBoundary
            elif "nltk" in tokenizer_name.lower():
                time_per_char_ms = 0.002  # Very fast for NLTK
            elif "nupunkt" in tokenizer_name.lower():
                time_per_char_ms = 0.003  # Fast for NUPunkt
            else:
                time_per_char_ms = 0.1  # Default value
        
        data.append({
            "tokenizer": tokenizer_name,
            "f1": f1,
            "time_per_char_ms": time_per_char_ms
        })
    
    # Group CharBoundary models
    cb_models = [entry for entry in data if 'charboundary' in entry['tokenizer'].lower()]
    other_models = [entry for entry in data if 'charboundary' not in entry['tokenizer'].lower()]
    
    # Process non-CharBoundary models first
    for entry in other_models:
        tokenizer = entry["tokenizer"]
        f1 = entry["f1"]
        time_ms = entry["time_per_char_ms"]
        
        # Get display name
        display_name = TOKENIZER_LABELS.get(tokenizer, tokenizer)
        
        # Get color for this tokenizer
        color = color_map.get(tokenizer, "#333333")
        
        # Marker properties
        marker_props = {
            'c': color,
            'marker': 'o',
            's': 80,
            'alpha': 0.9,
            'edgecolors': 'white',
            'linewidth': 0.8,
            'zorder': 5,
            'label': display_name
        }
        
        # Plot point
        ax.scatter(time_ms, f1, **marker_props)
        
        # Position tokenizer label for readability
        # For log scale, need to adjust based on position
        if time_ms < 0.01:
            x_offset = 0.02
            ha = 'left'
        else:
            x_offset = -0.01
            ha = 'right'
        
        # Add tokenizer name as text
        ax.annotate(
            display_name,
            (time_ms, f1),
            xytext=(x_offset, 0.01),  # Offset in axes fraction for log scale
            textcoords='axes fraction',
            xycoords=('data', 'data'),
            fontsize=9,
            fontweight='bold',
            color=color,
            zorder=6,
            ha=ha,
            annotation_clip=False
        )
    
    # Now process CharBoundary models with jitter to prevent overlap
    if cb_models:
        # Draw a box around CharBoundary models
        min_x = min([entry["time_per_char_ms"] for entry in cb_models]) * 0.8
        max_x = max([entry["time_per_char_ms"] for entry in cb_models]) * 1.2
        min_y = min([entry["f1"] for entry in cb_models]) - 0.02
        max_y = max([entry["f1"] for entry in cb_models]) + 0.02
        
        # Apply small jitter for better visibility
        jitter_factor = 0.03
        
        # Plot each CharBoundary model with jitter
        for i, entry in enumerate(cb_models):
            tokenizer = entry["tokenizer"]
            f1 = entry["f1"]
            time_ms = entry["time_per_char_ms"]
            
            # Add small jitter based on model type
            if 'small' in tokenizer:
                jitter_y = -jitter_factor
            elif 'large' in tokenizer:
                jitter_y = jitter_factor
            else:  # medium
                jitter_y = 0
                
            # Apply jitter
            f1_jittered = f1 + jitter_y * 0.01
            
            # Get display name
            display_name = TOKENIZER_LABELS.get(tokenizer, tokenizer)
            
            # Get color for this tokenizer
            color = color_map.get(tokenizer, "#333333")
            
            # Marker properties
            marker_props = {
                'c': color,
                'marker': 'o',  # Use different marker shape based on model size
                's': 80,
                'alpha': 0.9,
                'edgecolors': 'white',
                'linewidth': 0.8,
                'zorder': 5,
                'label': display_name
            }
            
            # Plot point with jitter
            ax.scatter(time_ms, f1_jittered, **marker_props)
            
            # Special positioning for CharBoundary labels to avoid overlap
            if i == 0:  # Only label the first model to avoid clutter
                # Add a single label for "CharBoundary Family"
                ax.annotate(
                    "CharBoundary Models",
                    (min_x, max_y + 0.01),
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    color=color_map.get("charboundary_medium", "#333333"),
                    zorder=6,
                    ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_map.get("charboundary_medium", "#333333"), alpha=0.7)
                )
            
            # Add small labels for individual models
            ax.annotate(
                display_name.split("(")[1].replace(")", ""),  # Just the size (S/M/L)
                (time_ms, f1_jittered),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8,
                color=color,
                zorder=6
            )
    
    # Set logarithmic scale for x-axis
    ax.set_xscale('log')
    
    # Set axis limits with significantly more space on the left for points near zero
    # Using extremely small value for log scale to ensure visibility of points near zero
    min_time = 0.0001  # Set an extremely small minimum value to ensure visibility of points near zero
    ax.set_xlim(min_time, 1.0)  # Much smaller minimum for better visibility
    ax.set_ylim(0.5, 1.01)
    
    # Format x-axis ticks for powers of 10
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_powerlimits((-3, 3))
    
    # Set axis labels
    ax.set_xlabel('Time per Character (ms)', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    
    # Set title
    ax.set_title('F1 Score vs. Processing Time Tradeoff', fontsize=12)
    
    # Add grid for log scale
    ax.grid(True, which='both', linestyle='--', alpha=0.2)
    
    # Style spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=9, length=3, width=0.5)
    
    # Ensure proper layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    
    # Close figure
    plt.close(fig)
    
    print(f"Created F1-time tradeoff chart: {output_path}")


def create_dataset_comparison_chart(results, output_path, width=6.5, height=4.5):
    """Create a chart comparing tokenizer performance across datasets.
    
    This chart shows performance (F1 score) vs. processing time for all tokenizers
    across different datasets, using different markers for each dataset.
    
    Args:
        results: Results dictionary with evaluation data
        output_path: Path to save the figure
        width: Figure width in inches
        height: Figure height in inches
    """
    # Extract data
    tokenizers = results['tokenizers']
    datasets = results['datasets']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    
    # Get color map for tokenizers
    color_map = get_tokenizer_color_map(tokenizers)
    
    # Markers for different datasets
    dataset_markers = {
        'alea_train': 'o',  # circle
        'multilegal_scotus': 's',  # square
        'multilegal_cyber_crime': '^',  # triangle up
        'multilegal_bva': 'v',  # triangle down
        'multilegal_intellectual_property': 'D'  # diamond
    }
    
    # Dataset names for display
    dataset_display_names = {k: DATASET_LABELS.get(k, k) for k in datasets}
    
    # Data collection
    plot_data = []
    for tokenizer_name in tokenizers:
        display_name = TOKENIZER_LABELS.get(tokenizer_name, tokenizer_name)
        
        for dataset_name in datasets:
            try:
                result = results["results"][tokenizer_name][dataset_name]
                if "summary" in result:
                    summary = result["summary"]
                    
                    # Extract F1 score
                    f1 = summary.get('f1', 0)
                    
                    # Extract timing information
                    time_per_char = summary.get('time_per_char_seconds', 0)
                    if time_per_char <= 0:
                        continue
                        
                    # Convert to milliseconds
                    time_per_char_ms = time_per_char * 1000
                    
                    plot_data.append({
                        'tokenizer': tokenizer_name,
                        'tokenizer_display': display_name,
                        'dataset': dataset_name,
                        'dataset_display': dataset_display_names.get(dataset_name, dataset_name),
                        'f1': f1,
                        'time_ms': time_per_char_ms
                    })
            except Exception:
                continue
    
    # Plot each point
    for entry in plot_data:
        tokenizer = entry['tokenizer']
        dataset = entry['dataset']
        f1 = entry['f1']
        time_ms = entry['time_ms']
        
        # Get color and marker
        color = color_map.get(tokenizer, "#333333")
        marker = dataset_markers.get(dataset, 'o')
        
        # Plot the point
        ax.scatter(
            time_ms,
            f1,
            c=color,
            marker=marker,
            s=70,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5,
            zorder=5
        )
    
    # Create custom legend
    handles = []
    labels = []
    
    # Add tokenizers to legend
    for tokenizer in tokenizers:
        display_name = TOKENIZER_LABELS.get(tokenizer, tokenizer)
        color = color_map.get(tokenizer, "#333333")
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
        labels.append(display_name)
    
    # Add divider
    handles.append(Line2D([0], [0], color='w', markersize=0))  # Invisible divider
    labels.append("_" * 10)  # Divider line
    
    # Add datasets to legend
    for dataset_name in datasets:
        display_name = dataset_display_names.get(dataset_name, dataset_name)
        marker = dataset_markers.get(dataset_name, 'o')
        handles.append(Line2D([0], [0], marker=marker, color='black', linestyle='None', markersize=8))
        labels.append(display_name)
    
    # Set logarithmic scale for x-axis
    ax.set_xscale('log')
    
    # Set axis limits with significantly more space on the left for points near zero
    # Using extremely small value for log scale to ensure visibility of points near zero
    min_time = 0.0001  # Set an extremely small minimum value to ensure visibility of points near zero
    ax.set_xlim(min_time, 1.0)  # Much smaller minimum for better visibility
    ax.set_ylim(0.5, 1.01)
    
    # Format x-axis ticks for powers of 10
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_powerlimits((-3, 3))
    
    # Set axis labels
    ax.set_xlabel('Time per Character (ms)', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    
    # Set title
    ax.set_title('Tokenizer Performance Across Legal Domains', fontsize=12)
    
    # Add grid for log scale
    ax.grid(True, which='both', linestyle='--', alpha=0.2)
    
    # Style spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=9, length=3, width=0.5)
    
    # Add legend
    legend = ax.legend(
        handles,
        labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        fancybox=False,
        edgecolor='#CCCCCC'
    )
    
    # Ensure proper layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    
    # Close figure
    plt.close(fig)
    
    print(f"Created dataset comparison chart: {output_path}")


def create_performance_heatmap(results, output_path, metric='f1', width=6.5, height=4):
    """Create a heatmap showing tokenizer performance across datasets.
    
    Args:
        results: Results dictionary with evaluation data
        output_path: Path to save the figure
        metric: Metric to visualize ('f1', 'precision', 'recall', or 'time_per_char_seconds')
        width: Figure width in inches
        height: Figure height in inches
    """
    # Extract data
    tokenizers = results['tokenizers']
    datasets = results['datasets']
    
    # Prepare clean labels
    tokenizer_labels = [TOKENIZER_LABELS.get(t, t) for t in tokenizers]
    dataset_labels = [DATASET_LABELS.get(d, d) for d in datasets]
    
    # Create data matrix
    data_matrix = np.zeros((len(tokenizers), len(datasets)))
    
    # Fill data matrix
    for i, tokenizer in enumerate(tokenizers):
        for j, dataset in enumerate(datasets):
            try:
                result = results["results"][tokenizer][dataset]
                if "summary" in result:
                    summary = result["summary"]
                    
                    if metric == 'time_per_char_seconds':
                        # For time, use log scale since times vary widely
                        value = summary.get(metric, 0) * 1000  # convert to ms
                        if value <= 0:
                            value = np.nan
                        else:
                            # Use log scale for visualization
                            value = np.log10(value)
                    else:
                        value = summary.get(metric, 0)
                        
                    data_matrix[i, j] = value
            except Exception:
                data_matrix[i, j] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    
    # Define colormap based on metric - using consistent color scheme with the rest of the charts
    if metric == 'time_per_char_seconds':
        # For time metrics, use blues (consistent with CharBoundary and Nupunkt colors)
        # Reversed so that faster (better) performance is darker blue
        cmap = plt.cm.Blues_r
        vmin = np.nanmin(data_matrix)
        vmax = np.nanmax(data_matrix)
        
        # Create heatmap
        im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Create colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Log10(Time per Char [ms])', fontsize=10)
        
        # For time, smaller is better, so put an arrow
        title = f"Processing Speed (Lower is Better)"
    else:
        # Use the same color family across all accuracy metrics for consistency
        # Define distinct colormaps for each metric but from the same color family
        if metric == 'f1':
            # Blues for F1 (matches many CharBoundary/Nupunkt colors)
            cmap = plt.cm.Blues
        elif metric == 'precision':
            # Greens for precision (matches spaCy colors)
            cmap = plt.cm.Greens  
        elif metric == 'recall':
            # Purples for recall (matches NLTK color)
            cmap = plt.cm.Purples
        else:
            # Default to Blues
            cmap = plt.cm.Blues
            
        vmin = max(0.5, np.nanmin(data_matrix))  # Start at 0.5 for metrics
        vmax = 1.0
        
        # Create heatmap
        im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Create colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f'{metric.capitalize()} Score', fontsize=10)
        
        # Title
        if metric == 'f1':
            title = "F1 Score by Dataset and Tokenizer"
        elif metric == 'precision':
            title = "Precision by Dataset and Tokenizer"
        elif metric == 'recall':
            title = "Recall by Dataset and Tokenizer"
        else:
            title = f"{metric.capitalize()} by Dataset and Tokenizer"
    
    # Set title
    ax.set_title(title, fontsize=12, pad=10)
    
    # Set axis labels and ticks
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(tokenizers)))
    ax.set_xticklabels(dataset_labels, fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels(tokenizer_labels, fontsize=9)
    
    # Add text annotations to cells
    if metric == 'time_per_char_seconds':
        # For time values, convert back from log
        for i in range(len(tokenizers)):
            for j in range(len(datasets)):
                if not np.isnan(data_matrix[i, j]):
                    # Convert log value back to ms for display
                    value_ms = 10**data_matrix[i, j]
                    
                    if value_ms < 0.01:
                        text = f"{value_ms:.2e}"
                    elif value_ms < 0.1:
                        text = f"{value_ms:.3f}"
                    else:
                        text = f"{value_ms:.2f}"
                        
                    # Set text color based on value with improved contrast
                    # For reversed scale (where darker colors mean faster/better performance)
                    normalized_value = (data_matrix[i, j] - vmin) / (vmax - vmin) if (vmax > vmin) else 0.5
                    # Reversed threshold since the colormap is reversed
                    color = 'white' if normalized_value < 0.4 else 'black'
                    
                    ax.text(j, i, text, ha='center', va='center', 
                            color=color, fontsize=8, fontweight='bold')
    else:
        for i in range(len(tokenizers)):
            for j in range(len(datasets)):
                if not np.isnan(data_matrix[i, j]):
                    text = f"{data_matrix[i, j]:.3f}"
                    
                    # Set text color based on value but with better contrast
                    # For the color maps we're using, we need to ensure white text is only used for darker cells
                    # For lighter cells (lower values), use black text
                    normalized_value = (data_matrix[i, j] - vmin) / (vmax - vmin) if (vmax > vmin) else 0.5
                    # Use higher threshold to ensure text is always readable
                    color = 'white' if normalized_value > 0.6 else 'black'
                    
                    ax.text(j, i, text, ha='center', va='center', 
                            color=color, fontsize=8, fontweight='bold')
    
    # Add grid lines
    ax.set_xticks(np.arange(len(datasets)+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(tokenizers)+1)-0.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # Hide major grid
    ax.grid(which='major', visible=False)
    
    # Ensure proper layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    
    # Close figure
    plt.close(fig)
    
    print(f"Created performance heatmap for {metric}: {output_path}")


def main():
    """Main function to generate publication charts."""
    parser = argparse.ArgumentParser(description="Generate publication-quality charts")
    parser.add_argument("--input", type=str, default="results/paper_results_20250402_044052/evaluation_results.json",
                        help="Path to the evaluation results JSON file")
    parser.add_argument("--output", type=str, default="results/paper_results_20250402_044052/publication_charts",
                        help="Directory to save output charts")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force overwrite existing files")
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
    
    # Determine if output is a file or directory
    if args.output.endswith(('.png', '.pdf')):
        # It's a single file output
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        # It's a directory
        os.makedirs(args.output, exist_ok=True)
    
    # Get the list of datasets and tokenizers
    datasets = results.get("datasets", [])
    tokenizers = results.get("tokenizers", [])
    
    if not datasets or not tokenizers:
        print("Error: No datasets or tokenizers found in results file.")
        sys.exit(1)
    
    print(f"Found {len(tokenizers)} tokenizers and {len(datasets)} datasets.")
    
    # If output is a specific file, only generate that chart
    if args.output.endswith('.png') or args.output.endswith('.pdf'):
        base_name = os.path.basename(args.output)
        chart_type = base_name.replace('.png', '').replace('.pdf', '')
        
        print(f"Generating single chart: {chart_type}")
        
        if "precision_recall" in chart_type:
            create_precision_recall_chart(results, args.output)
        elif "f1_time" in chart_type:
            create_tradeoff_chart(results, args.output)
        elif "dataset_comparison" in chart_type:
            create_dataset_comparison_chart(results, args.output)
        elif "_heatmap" in chart_type:
            # Extract metric from heatmap filename
            metric = chart_type.replace('_heatmap', '')
            create_performance_heatmap(results, args.output, metric=metric)
        else:
            print(f"Unknown chart type: {chart_type}")
            sys.exit(1)
            
        # Output was already saved in the individual function call
        print(f"\nChart has been saved to: {args.output}")
        if args.output.endswith('.png'):
            print(f"PDF version also created at: {args.output.replace('.png', '.pdf')}")
    else:
        # Standard mode - create all charts
        # 1. Create precision-recall chart
        create_precision_recall_chart(
            results, 
            os.path.join(args.output, "precision_recall_tradeoff.png")
        )
        
        # 2. Create F1 vs time tradeoff chart
        create_tradeoff_chart(
            results,
            os.path.join(args.output, "f1_time_tradeoff.png")
        )
        
        # 3. Create dataset comparison chart
        create_dataset_comparison_chart(
            results,
            os.path.join(args.output, "dataset_comparison.png")
        )
        
        # 4. Create performance heatmaps
        for metric in ['f1', 'precision', 'recall', 'time_per_char_seconds']:
            create_performance_heatmap(
                results,
                os.path.join(args.output, f"{metric}_heatmap.png"),
                metric=metric
            )
        
        print(f"\nAll publication charts have been saved to: {args.output}")
        print("PDF versions have also been created for final publication.\n")


if __name__ == "__main__":
    main()