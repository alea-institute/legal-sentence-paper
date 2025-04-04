"""Evaluation utilities for sentence boundary detection."""

import json
import os
import time
import warnings
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Filter out spaCy's lemmatizer warnings
warnings.filterwarnings("ignore", message=".*W108.*lemmatizer did not find POS annotation.*")

from lsp.core.tokenizer import SentenceTokenizer
from lsp.core.data_loader import DataLoader
from lsp.visualization_html import generate_html_report


class Evaluator:
    """Evaluate sentence tokenizer performance on datasets."""

    def __init__(self):
        """Initialize the evaluator."""
        self.tokenizers = {}
        self.datasets = {}
        self.results = {}

    def add_tokenizer(self, tokenizer: SentenceTokenizer) -> None:
        """Add a tokenizer for evaluation.
        
        Args:
            tokenizer: An initialized SentenceTokenizer instance
            
        Raises:
            ValueError: If the tokenizer is not initialized
        """
        if not tokenizer.is_initialized:
            raise ValueError(f"Tokenizer {tokenizer.name} must be initialized before adding")
        
        self.tokenizers[tokenizer.name] = tokenizer

    def add_dataset(self, dataset: DataLoader) -> None:
        """Add a dataset for evaluation.
        
        Args:
            dataset: A loaded DataLoader instance
            
        Raises:
            ValueError: If the dataset is not loaded
        """
        if not dataset.is_loaded:
            raise ValueError(f"Dataset {dataset.name} must be loaded before adding")
        
        self.datasets[dataset.name] = dataset

    def evaluate_all(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate all tokenizers on all datasets.
        
        Args:
            sample_size: Optional number of examples to sample from each dataset
                        If None, uses all examples
                        
        Returns:
            Dictionary of evaluation results
        """
        if not self.tokenizers:
            raise ValueError("No tokenizers added for evaluation")
        
        if not self.datasets:
            raise ValueError("No datasets added for evaluation")
        
        all_results = {
            "tokenizers": list(self.tokenizers.keys()),
            "datasets": list(self.datasets.keys()),
            "sample_size": sample_size,
            "timestamp": time.time(),
            "results": {}
        }
        
        for tokenizer_name, tokenizer in self.tokenizers.items():
            print(f"\nEvaluating tokenizer: {tokenizer_name}")
            tokenizer_results = {}
            
            for dataset_name, dataset in self.datasets.items():
                print(f"  On dataset: {dataset_name} ({len(dataset)} examples)")
                
                # Sample examples if requested
                example_indices = None
                if sample_size and sample_size < len(dataset):
                    import random
                    example_indices = random.sample(range(len(dataset)), sample_size)
                    print(f"    Using random sample of {sample_size} examples")
                
                # Evaluate tokenizer on dataset
                start_time = time.time()
                result = dataset.evaluate_tokenizer(tokenizer, example_indices)
                end_time = time.time()
                
                # Add evaluation time
                result["eval_time_seconds"] = end_time - start_time
                
                # Store result
                tokenizer_results[dataset_name] = result
                
                # Print summary
                summary = result["summary"]
                print(f"    Precision: {summary['precision']:.4f}")
                print(f"    Recall: {summary['recall']:.4f}")
                print(f"    F1: {summary['f1']:.4f}")
                print(f"    Time per char: {summary['time_per_char_seconds'] * 1000:.4f} ms")
                print(f"    Time per sentence: {summary['time_per_sentence_seconds'] * 1000:.4f} ms")
            
            all_results["results"][tokenizer_name] = tokenizer_results
        
        # Store results
        self.results = all_results
        return all_results

    def save_results(self, output_path: str) -> None:
        """Save evaluation results to a file.
        
        Args:
            output_path: Path to save the results
        """
        if not self.results:
            raise ValueError("No evaluation results to save")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\nResults saved to {output_path}")
        
    def generate_html_report(self, output_path: str, max_examples_per_dataset: int = 10) -> None:
        """Generate an HTML report with color-coded sentence visualizations.
        
        Args:
            output_path: Path to save the HTML report
            max_examples_per_dataset: Maximum number of examples to include per dataset
        """
        if not self.results:
            raise ValueError("No evaluation results to visualize")
            
        try:
            # Use the visualization_html module to generate the report
            generate_html_report(self.results, output_path, max_examples_per_dataset)
            print(f"\nHTML report generated at {output_path}")
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            import traceback
            traceback.print_exc()

    def print_summary_table(self) -> None:
        """Print a summary table of the evaluation results."""
        if not self.results:
            raise ValueError("No evaluation results to print")
        
        print("\n===== Tokenizer Performance Summary =====")
        
        # Prepare table data
        table_data = []
        
        for tokenizer_name in self.results["tokenizers"]:
            for dataset_name in self.results["datasets"]:
                result = self.results["results"][tokenizer_name][dataset_name]
                summary = result["summary"]
                
                table_data.append([
                    tokenizer_name,
                    dataset_name,
                    f"{summary['precision']:.4f}",
                    f"{summary['recall']:.4f}",
                    f"{summary['f1']:.4f}",
                    f"{summary['accuracy']:.4f}",
                    f"{summary['time_per_char_seconds'] * 1000:.4f}",
                    f"{summary['time_per_sentence_seconds'] * 1000:.4f}",
                    summary['total_sentences']
                ])
        
        # Sort by F1 score
        table_data.sort(key=lambda x: (x[1], -float(x[4])))
        
        # Print table
        headers = [
            "Tokenizer", "Dataset", "Precision", "Recall", "F1", "Accuracy", 
            "Time/Char (ms)", "Time/Sent (ms)", "Sentences"
        ]
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def generate_comparison_chart(self, metric: str = "f1", output_path: Optional[str] = None) -> None:
        """Generate a comparison chart of the evaluation results.
        
        Args:
            metric: Metric to compare ('precision', 'recall', 'f1', 'accuracy',
                   'time_per_char_seconds', 'time_per_sentence_seconds')
            output_path: Optional path to save the chart
        """
        if not self.results:
            raise ValueError("No evaluation results to chart")
            
        # Check for required libraries
        try:
            import matplotlib.pyplot as plt
            from lsp.visualization import plot_tokenizer_comparison
        except ImportError as e:
            print(f"Required visualization libraries not available: {e}")
            print("Please install pandas, matplotlib, and numpy to generate charts")
            return
        
        # Valid metrics
        valid_metrics = [
            "precision", "recall", "f1", "accuracy", 
            "time_per_char_seconds", "time_per_sentence_seconds"
        ]
        
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")
        
        # Use the visualization module to create the chart
        try:
            fig, ax = plot_tokenizer_comparison(self.results, metric, output_path)
            
            # Save chart if requested and not already done by plot_tokenizer_comparison
            if output_path and not os.path.exists(output_path):
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Chart saved to {output_path}")
            
            # Close the figure to prevent display in notebooks
            plt.close(fig)
            
        except Exception as e:
            print(f"Error generating chart: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def confusion_matrix(results: Dict[str, Any], tokenizer_name: str, dataset_name: str) -> Dict[str, Any]:
        """Generate confusion matrix for a specific tokenizer and dataset.
        
        Args:
            results: Evaluation results dictionary
            tokenizer_name: Name of the tokenizer
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with confusion matrix and metrics
        """
        # Check if results exist
        if not results or "results" not in results:
            raise ValueError("Invalid results dictionary")
            
        if tokenizer_name not in results.get("results", {}):
            raise ValueError(f"Tokenizer {tokenizer_name} not found in results")
            
        if dataset_name not in results["results"][tokenizer_name]:
            raise ValueError(f"Dataset {dataset_name} not found in results for tokenizer {tokenizer_name}")
        
        # Get result
        result = results["results"][tokenizer_name][dataset_name]
        summary = result["summary"]
        
        # Confusion matrix values
        tp = summary["total_true_positives"]
        fp = summary["total_false_positives"]
        fn = summary["total_false_negatives"]
        
        # Calculate true negatives (not directly available)
        # For sentence boundaries, we don't have a clear notion of true negatives
        # We could consider total characters minus the boundaries as a proxy
        tn = 0
        
        # Calculate metrics
        precision = summary["precision"]
        recall = summary["recall"]
        f1 = summary["f1"]
        accuracy = summary["accuracy"]
        
        return {
            "matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn
            },
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy
            }
        }


def initialize_tokenizers() -> Dict[str, SentenceTokenizer]:
    """Initialize all available tokenizers.
    
    Returns:
        Dictionary of tokenizer name to initialized tokenizer
    """
    from lsp.tokenizers.nupunkt import NupunktTokenizer
    from lsp.tokenizers.nltk import NLTKTokenizer
    from lsp.tokenizers.spacy import SpacyTokenizer
    from lsp.tokenizers.pysbd import PySBDTokenizer
    from lsp.tokenizers.charboundary import CharBoundaryTokenizer
    
    tokenizers = {}
    
    print("Initializing tokenizers...")
    
    # Initialize nupunkt
    try:
        print("  - nupunkt")
        nupunkt = NupunktTokenizer()
        nupunkt.initialize()
        tokenizers[nupunkt.name] = nupunkt
    except Exception as e:
        print(f"    Failed to initialize nupunkt: {e}")
    
    # Initialize NLTK
    try:
        print("  - nltk_punkt")
        nltk = NLTKTokenizer()
        nltk.initialize()
        tokenizers[nltk.name] = nltk
    except Exception as e:
        print(f"    Failed to initialize NLTK: {e}")
    
    # Initialize spaCy - small model
    try:
        print("  - spacy (en_core_web_sm)")
        spacy_sm = SpacyTokenizer(name="spacy_sm")
        spacy_sm.initialize(model="en_core_web_sm")
        tokenizers[spacy_sm.name] = spacy_sm
    except Exception as e:
        print(f"    Failed to initialize spaCy small model: {e}")
    
    # Initialize spaCy - large model (optional)
    try:
        print("  - spacy (en_core_web_lg)")
        spacy_lg = SpacyTokenizer(name="spacy_lg")
        spacy_lg.initialize(model="en_core_web_lg")
        tokenizers[spacy_lg.name] = spacy_lg
    except Exception as e:
        print(f"    Note: spaCy large model could not be initialized: {e}")
    
    # Initialize PySBD
    try:
        print("  - pysbd")
        pysbd = PySBDTokenizer()
        pysbd.initialize()
        tokenizers[pysbd.name] = pysbd
    except Exception as e:
        print(f"    Failed to initialize PySBD: {e}")
    
    # Initialize CharBoundary models
    try:
        print("  - charboundary (small)")
        cb_small = CharBoundaryTokenizer(name="charboundary_small")
        cb_small.initialize(size="small")
        tokenizers[cb_small.name] = cb_small
    except Exception as e:
        print(f"    Failed to initialize CharBoundary small: {e}")
    
    try:
        print("  - charboundary (medium)")
        cb_medium = CharBoundaryTokenizer(name="charboundary_medium")
        cb_medium.initialize(size="medium")
        tokenizers[cb_medium.name] = cb_medium
    except Exception as e:
        print(f"    Failed to initialize CharBoundary medium: {e}")
    
    try:
        print("  - charboundary (large)")
        cb_large = CharBoundaryTokenizer(name="charboundary_large")
        cb_large.initialize(size="large")
        tokenizers[cb_large.name] = cb_large
    except Exception as e:
        print(f"    Failed to initialize CharBoundary large: {e}")
    
    print(f"Successfully initialized {len(tokenizers)} tokenizers")
    return tokenizers


def load_datasets(limit: Optional[int] = None) -> Dict[str, DataLoader]:
    """Load all available datasets.
    
    Args:
        limit: Optional maximum number of examples to load per dataset
        
    Returns:
        Dictionary of dataset name to loaded dataset
    """
    from lsp.dataloaders.discovery import load_all_datasets
    
    print("Loading datasets...")
    
    # Use the discovery mechanism to find and load all datasets
    datasets = load_all_datasets(limit=limit)
    
    print(f"Successfully loaded {len(datasets)} datasets")
    return datasets


def main():
    """Main function to run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate sentence tokenizers on legal datasets")
    parser.add_argument("--sample", type=int, help="Number of examples to sample for evaluation")
    parser.add_argument("--load-limit", type=int, help="Limit number of examples to load from each dataset")
    parser.add_argument("--output", type=str, default="results/evaluation_results.json", help="Path to save results")
    parser.add_argument("--charts", action="store_true", help="Generate comparison charts")
    parser.add_argument("--html", action="store_true", help="Generate HTML report with sentence visualizations")
    parser.add_argument("--html-output", type=str, default="results/evaluation_report.html", help="Path to save HTML report")
    parser.add_argument("--max-examples", type=int, default=10, help="Maximum number of examples per dataset in HTML report")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")
    args = parser.parse_args()
    
    # Initialize tokenizers
    tokenizers = initialize_tokenizers()
    
    # Load datasets (with limit if specified)
    datasets = load_datasets(limit=args.load_limit)
    
    if not tokenizers:
        print("Error: No tokenizers could be initialized. Aborting.")
        return
        
    if not datasets:
        print("Error: No datasets could be loaded. Aborting.")
        return
    
    # Create evaluator
    evaluator = Evaluator()
    
    # Add tokenizers and datasets
    for tokenizer in tokenizers.values():
        evaluator.add_tokenizer(tokenizer)
    
    for dataset in datasets.values():
        evaluator.add_dataset(dataset)
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate_all(sample_size=args.sample)
    
    # Print summary
    evaluator.print_summary_table()
    
    # Save results
    if not args.no_save:
        # Create directories if needed
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        evaluator.save_results(args.output)
    
    # Generate HTML report if requested
    if args.html:
        print("\nGenerating HTML report with sentence visualizations...")
        os.makedirs(os.path.dirname(args.html_output), exist_ok=True)
        evaluator.generate_html_report(args.html_output, max_examples_per_dataset=args.max_examples)
    
    # Generate charts if requested
    if args.charts:
        print("\nGenerating charts...")
        
        # Create chart directories
        charts_base_dir = "results/charts"
        os.makedirs(f"{charts_base_dir}/comparison", exist_ok=True)
        os.makedirs(f"{charts_base_dir}/tradeoffs", exist_ok=True) 
        os.makedirs(f"{charts_base_dir}/weighted", exist_ok=True)
        
        # Import additional visualization functions
        try:
            from lsp.visualization import (
                plot_weighted_metrics,
                plot_precision_recall_tradeoff,
                plot_performance_vs_throughput,
                plot_custom_tradeoff_chart
            )
            
            # 1. Standard comparison charts for each metric
            print("  - Generating comparison charts...")
            for metric in ["f1", "precision", "recall", "time_per_char_seconds", "time_per_sentence_seconds"]:
                metric_name = metric.replace("_", "-")
                output_path = f"{charts_base_dir}/comparison/{metric_name}.png"
                
                try:
                    evaluator.generate_comparison_chart(metric=metric, output_path=output_path)
                except Exception as e:
                    print(f"    Error generating chart for {metric}: {e}")
            
            # 2. Sample-size weighted metrics chart
            try:
                print("  - Generating weighted metrics chart...")
                weighted_output_path = f"{charts_base_dir}/weighted/weighted_metrics.png"
                fig, ax = plot_weighted_metrics(
                    evaluator.results,
                    output_path=weighted_output_path
                )
                plt.close(fig)
            except Exception as e:
                print(f"    Error generating weighted metrics chart: {e}")
            
            # 3. Precision-recall tradeoff scatter plot
            try:
                print("  - Generating precision-recall tradeoff chart...")
                pr_output_path = f"{charts_base_dir}/tradeoffs/precision_recall.png"
                fig, ax = plot_precision_recall_tradeoff(
                    evaluator.results,
                    output_path=pr_output_path
                )
                plt.close(fig)
            except Exception as e:
                print(f"    Error generating precision-recall tradeoff chart: {e}")
            
            # 4. Performance vs throughput/tradeoff charts
            print("  - Generating performance vs throughput charts...")
            
            # Use the new custom tradeoff chart for these visualizations
            for metric_name in ["precision", "recall", "f1"]:
                try:
                    # Standard throughput chart (x-axis is throughput)
                    output_path = f"{charts_base_dir}/tradeoffs/{metric_name}_vs_throughput.png"
                    fig, ax = plot_performance_vs_throughput(
                        evaluator.results,
                        metric=metric_name,
                        output_path=output_path
                    )
                    plt.close(fig)
                    
                    # Custom tradeoff chart (x-axis is time, inverted to show performance/cost tradeoff)
                    custom_output_path = f"{charts_base_dir}/tradeoffs/{metric_name}_time_tradeoff.png"
                    fig, ax = plot_custom_tradeoff_chart(
                        evaluator.results,
                        x_metric="time_per_char_seconds",
                        y_metric=metric_name,
                        output_path=custom_output_path,
                        aggregate_method="weighted",
                        invert_x=True,
                        title=f"{metric_name.title()} vs Performance Cost Tradeoff"
                    )
                    plt.close(fig)
                except Exception as e:
                    print(f"    Error generating {metric_name} vs throughput chart: {e}")
                    
        except ImportError as e:
            print(f"Could not import all visualization functions: {e}")
            print("Some advanced charts will not be generated.")


if __name__ == "__main__":
    main()