"""
Run complete evaluation of all tokenizers on all datasets.

This script initializes all tokenizers, loads all datasets,
and runs evaluation to generate comprehensive results.
"""

import os
import sys
import argparse
import time
import warnings
from typing import Dict, List, Any, Optional

# Filter out spaCy's lemmatizer warnings
warnings.filterwarnings("ignore", message=".*W108.*lemmatizer did not find POS annotation.*")

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lsp.evaluation import Evaluator, initialize_tokenizers, load_datasets


def run_evaluation(sample_size: Optional[int] = None,
                  output_dir: str = "results",
                  generate_charts: bool = True) -> None:
    """Run evaluation of all tokenizers on all datasets.
    
    Args:
        sample_size: Optional number of examples to sample from each dataset
        output_dir: Directory to save results
        generate_charts: Whether to generate comparison charts
    """
    print("=== Legal Sentence Boundary Detection Evaluation ===\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if generate_charts:
        os.makedirs(os.path.join(output_dir, "charts"), exist_ok=True)
    
    # Initialize tokenizers
    tokenizers = initialize_tokenizers()
    if not tokenizers:
        print("Error: No tokenizers could be initialized. Aborting.")
        return
    
    # Load datasets (pass sample_size as limit if provided)
    dataset_limit = sample_size if sample_size else None
    datasets = load_datasets(limit=dataset_limit)
    if not datasets:
        print("Error: No datasets could be loaded. Aborting.")
        return
    
    # Create evaluator
    evaluator = Evaluator()
    
    # Add tokenizers and datasets
    for name, tokenizer in tokenizers.items():
        evaluator.add_tokenizer(tokenizer)
        
    for name, dataset in datasets.items():
        evaluator.add_dataset(dataset)
    
    # Run evaluation
    print("\n=== Running Evaluation ===")
    start_time = time.time()
    
    results = evaluator.evaluate_all(sample_size=sample_size)
    
    end_time = time.time()
    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")
    
    # Print summary
    evaluator.print_summary_table()
    
    # Save results
    output_path = os.path.join(output_dir, "evaluation_results.json")
    evaluator.save_results(output_path)
    
    # Generate charts if requested
    if generate_charts:
        print("\n=== Generating Comparison Charts ===")
        
        metrics = [
            ("f1", "F1 Score"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("accuracy", "Accuracy"),
            ("time_per_char_seconds", "Time per Character (ms)"),
            ("time_per_sentence_seconds", "Time per Sentence (ms)")
        ]
        
        for metric, title in metrics:
            metric_name = metric.replace("_", "-")
            output_path = os.path.join(output_dir, "charts", f"comparison_{metric_name}.png")
            
            try:
                print(f"Generating chart for {title}...")
                evaluator.generate_comparison_chart(metric=metric, output_path=output_path)
            except Exception as e:
                print(f"Error generating chart for {metric}: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate sentence tokenizers on legal datasets")
    parser.add_argument("--sample", type=int, help="Number of examples to sample from each dataset")
    parser.add_argument("--output", type=str, default="results", help="Directory to save results")
    parser.add_argument("--no-charts", action="store_true", help="Skip generating comparison charts")
    args = parser.parse_args()
    
    run_evaluation(
        sample_size=args.sample,
        output_dir=args.output,
        generate_charts=not args.no_charts
    )


if __name__ == "__main__":
    main()