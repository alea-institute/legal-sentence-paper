"""
Complete workflow for paper experiments.

This script runs the complete analysis workflow for the legal sentence paper:
1. Initialize all tokenizers
2. Load all datasets
3. Calculate dataset statistics
4. Run tokenizer evaluation
5. Generate comparison visualizations
6. Output results in a paper-ready format
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Filter out spaCy's lemmatizer warnings
warnings.filterwarnings("ignore", message=".*W108.*lemmatizer did not find POS annotation.*")

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import key components
from lsp.evaluation import Evaluator
from lsp.dataloaders.discovery import load_all_datasets
from lsp.tokenizers.nupunkt import NupunktTokenizer
from lsp.tokenizers.nltk import NLTKTokenizer
from lsp.tokenizers.spacy import SpacyTokenizer
from lsp.tokenizers.pysbd import PySBDTokenizer
from lsp.tokenizers.charboundary import CharBoundaryTokenizer


def initialize_all_tokenizers() -> Dict[str, Any]:
    """Initialize all tokenizers for the paper experiments.
    
    Returns:
        Dictionary of tokenizer name to tokenizer instance
    """
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
        print("  - spacy_sm")
        spacy_sm = SpacyTokenizer(name="spacy_sm")
        spacy_sm.initialize(model="en_core_web_sm")
        tokenizers[spacy_sm.name] = spacy_sm
    except Exception as e:
        print(f"    Failed to initialize spaCy small model: {e}")
    
    # Initialize spaCy - large model (optional)
    try:
        print("  - spacy_lg")
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
    
    # Initialize CharBoundary models - all three sizes
    for size in ["small", "medium", "large"]:
        try:
            name = f"charboundary_{size}"
            print(f"  - {name}")
            cb = CharBoundaryTokenizer(name=name)
            cb.initialize(size=size)
            tokenizers[cb.name] = cb
        except Exception as e:
            print(f"    Failed to initialize CharBoundary {size}: {e}")
    
    print(f"Successfully initialized {len(tokenizers)} tokenizers")
    return tokenizers


def generate_latex_table(results: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Generate LaTeX table from evaluation results.
    
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
    latex = """\\begin{table}[htbp]
\\centering
\\caption{Sentence Boundary Detection Performance on Legal Texts}
\\label{tab:sbd-performance}
\\begin{tabular}{llrrrr}
\\toprule
Dataset & Model & Precision & Recall & F1 & Time (ms/char) \\\\
\\midrule
"""
    
    # Tokenizer metric data
    for dataset in datasets:
        first_dataset = True
        
        # Sort tokenizers by F1 score for this dataset
        sorted_tokenizers = sorted(
            tokenizers,
            key=lambda t: results["results"][t][dataset]["summary"]["f1"],
            reverse=True
        )
        
        for tokenizer in sorted_tokenizers:
            summary = results["results"][tokenizer][dataset]["summary"]
            
            # Format dataset name (only for first row)
            if first_dataset:
                dataset_name = dataset
                first_dataset = False
            else:
                dataset_name = ""
            
            # Format time in milliseconds
            time_ms = summary["time_per_char_seconds"] * 1000
            
            # Add row
            latex += f"{dataset_name} & {tokenizer} & {summary['precision']:.3f} & {summary['recall']:.3f} & {summary['f1']:.3f} & {time_ms:.3f} \\\\\n"
        
        # Add a midrule between datasets
        if dataset != datasets[-1]:
            latex += "\\midrule\n"
    
    # Close the LaTeX table
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Save if requested
    if output_path:
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to {output_path}")
    
    return latex


def run_workflow(args: argparse.Namespace) -> None:
    """Run the complete paper workflow.
    
    Args:
        args: Command line arguments
    """
    # Create timestamp for result directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output) / f"paper_results_{timestamp}"
    
    # Create results directory structure
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir / "charts", exist_ok=True)
    os.makedirs(results_dir / "stats", exist_ok=True)
    os.makedirs(results_dir / "latex", exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    
    # Step 1: Initialize all tokenizers
    print("\n=== Step 1: Initialize Tokenizers ===")
    tokenizers = initialize_all_tokenizers()
    
    if not tokenizers:
        print("Error: No tokenizers could be initialized. Aborting.")
        return
    
    # Save tokenizer info
    tokenizer_info = {
        name: {
            "name": tok.name,
            "initialization_time": tok.initialization_time,
            "initialization_options": tok.initialization_options
        }
        for name, tok in tokenizers.items()
    }
    
    with open(results_dir / "tokenizers.json", "w") as f:
        json.dump(tokenizer_info, f, indent=2)
    
    # Step 2: Load all datasets
    print("\n=== Step 2: Load Datasets ===")
    datasets = load_all_datasets(limit=args.sample)
    
    if not datasets:
        print("Error: No datasets could be loaded. Aborting.")
        return
    
    # Step 3: Calculate dataset statistics
    print("\n=== Step 3: Calculate Dataset Statistics ===")
    
    dataset_stats = {}
    for name, dataset in datasets.items():
        print(f"Analyzing dataset: {name}")
        stats = dataset.calculate_statistics()
        dataset_stats[name] = stats
        
        # Save individual dataset statistics
        with open(results_dir / "stats" / f"{name}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
    
    # Save combined dataset statistics
    with open(results_dir / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)
    
    # Step 4: Run tokenizer evaluation
    print("\n=== Step 4: Run Tokenizer Evaluation ===")
    
    # Create evaluator
    evaluator = Evaluator()
    
    # Add tokenizers and datasets
    for name, tokenizer in tokenizers.items():
        evaluator.add_tokenizer(tokenizer)
    
    for name, dataset in datasets.items():
        evaluator.add_dataset(dataset)
    
    # Run evaluation
    print("Running evaluation...")
    start_time = time.time()
    results = evaluator.evaluate_all(sample_size=args.eval_sample)
    end_time = time.time()
    
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")
    
    # Print summary
    evaluator.print_summary_table()
    
    # Save results
    evaluator.save_results(str(results_dir / "evaluation_results.json"))
    
    # We'll generate the HTML report AFTER creating all charts to ensure they're included
    
    # Step 5: Generate comparison visualizations
    print("\n=== Step 5: Generate Comparison Visualizations ===")
    
    # Import visualization libraries
    try:
        import matplotlib.pyplot as plt
        from lsp.visualization import (
            plot_tokenizer_comparison,
            plot_weighted_metrics,
            plot_precision_recall_tradeoff,
            plot_performance_vs_throughput,
            plot_custom_tradeoff_chart,
            create_dataset_specific_chart
        )
    except ImportError:
        print("Warning: matplotlib and numpy are required for visualizations")
        has_viz_libs = False
    else:
        has_viz_libs = True
    
    if has_viz_libs:
        # Create charts directory structure
        os.makedirs(results_dir / "charts" / "comparison", exist_ok=True)
        os.makedirs(results_dir / "charts" / "tradeoffs", exist_ok=True)
        os.makedirs(results_dir / "charts" / "weighted", exist_ok=True)
        
        # 1. Standard comparison charts for each metric
        metrics = [
            ("f1", "F1 Score"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("time_per_char_seconds", "Time per Character (ms)"),
            ("time_per_sentence_seconds", "Time per Sentence (ms)")
        ]
        
        print("\nGenerating comparison charts...")
        for metric, title in metrics:
            try:
                metric_name = metric.replace("_", "-")
                output_path = str(results_dir / "charts" / "comparison" / f"{metric_name}.png")
                
                print(f"  - {title} comparison...")
                # Create standard comparison chart
                fig, ax = plot_tokenizer_comparison(
                    evaluator.results, 
                    metric=metric, 
                    output_path=output_path
                )
                plt.close(fig)  # Close to prevent display
                
            except Exception as e:
                print(f"Error generating chart for {metric}: {e}")
        
        # 2. Sample-size weighted metrics chart
        try:
            print("\nGenerating weighted metrics chart...")
            weighted_output_path = str(results_dir / "charts" / "weighted" / "weighted_metrics.png")
            fig, ax = plot_weighted_metrics(
                evaluator.results,
                output_path=weighted_output_path
            )
            plt.close(fig)
        except Exception as e:
            print(f"Error generating weighted metrics chart: {e}")
        
        # Generate tradeoff charts
        try:
            print("\nGenerating tradeoff charts...")
            
            # 1. Precision vs Time per character
            # Note: Using "precision_vs_throughput.png" for backwards compatibility
            precision_vs_time_path = str(results_dir / "charts" / "tradeoffs" / "precision_vs_throughput.png")
            fig, ax = plot_custom_tradeoff_chart(
                evaluator.results,
                x_metric="time_per_char_seconds",
                y_metric="precision",
                output_path=precision_vs_time_path,
                aggregate_method="combined",  # Use combined raw counts
                invert_x=True,
                title="Precision vs Time per Character Tradeoff (Combined)"
            )
            plt.close(fig)
            
            # 2. F1 vs Time per character
            f1_vs_time_path = str(results_dir / "charts" / "tradeoffs" / "f1_vs_throughput.png")
            fig, ax = plot_custom_tradeoff_chart(
                evaluator.results,
                x_metric="time_per_char_seconds",
                y_metric="f1",
                output_path=f1_vs_time_path,
                aggregate_method="combined",  # Use combined raw counts
                invert_x=True,
                title="F1 Score vs Time per Character Tradeoff (Combined)"
            )
            plt.close(fig)
            
            # 3. Recall vs Time per character
            recall_vs_time_path = str(results_dir / "charts" / "tradeoffs" / "recall_vs_throughput.png")
            fig, ax = plot_custom_tradeoff_chart(
                evaluator.results,
                x_metric="time_per_char_seconds",
                y_metric="recall",
                output_path=recall_vs_time_path,
                aggregate_method="combined",  # Use combined raw counts
                invert_x=True,
                title="Recall vs Time per Character Tradeoff (Combined)"
            )
            plt.close(fig)
            
            # 4. Classic precision-recall tradeoff (already uses combined raw counts by default)
            precision_recall_path = str(results_dir / "charts" / "tradeoffs" / "precision_recall.png")
            fig, ax = plot_precision_recall_tradeoff(
                evaluator.results,
                output_path=precision_recall_path
            )
            plt.close(fig)
            
            # 5. Create per-dataset tradeoff charts (no combined counts needed since single dataset)
            print("  - Generating per-dataset tradeoff charts...")
            os.makedirs(results_dir / "charts" / "datasets", exist_ok=True)
            
            for dataset_name in evaluator.results["datasets"]:
                # Create performance vs time charts for each dataset
                dataset_output_path = str(results_dir / "charts" / "datasets" / f"{dataset_name}_f1_vs_time.png")
                fig, ax = create_dataset_specific_chart(
                    evaluator.results,
                    dataset_name=dataset_name,
                    x_metric="time_per_char_seconds",
                    y_metric="f1",
                    output_path=dataset_output_path,
                    title=f"F1 vs Performance for {dataset_name}",
                    invert_x=True
                )
                plt.close(fig)
            
        except Exception as e:
            print(f"Error generating tradeoff charts: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 6: Generate paper-ready LaTeX tables
    print("\n=== Step 6: Generate LaTeX Tables ===")
    
    # Generate main performance table
    latex_table = generate_latex_table(
        results, 
        output_path=str(results_dir / "latex" / "performance_table.tex")
    )
    
    # Generate dataset statistics table
    latex_stats = """\\begin{table}[htbp]
\\centering
\\caption{Legal Dataset Statistics}
\\label{tab:dataset-stats}
\\begin{tabular}{lrrrr}
\\toprule
Dataset & Examples & Sentences & Avg. Sentences/Doc & Avg. Sentence Length \\\\
\\midrule
"""
    
    for name, stats in dataset_stats.items():
        latex_stats += f"{name} & {stats['total_examples']} & {stats['total_sentences']} & {stats['avg_sentences_per_example']:.1f} & {stats['avg_sentence_length']:.1f} \\\\\n"
    
    latex_stats += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open(results_dir / "latex" / "dataset_stats_table.tex", "w") as f:
        f.write(latex_stats)
        
    # Step 7: Generate publication-quality charts
    print("\n=== Step 7: Generate Publication-Quality Charts ===")
    
    try:
        # Check if the publication_charts module exists
        from lsp.examples.publication_charts import (
            create_precision_recall_chart,
            create_tradeoff_chart,
            create_dataset_comparison_chart,
            create_performance_heatmap
        )
        
        # Create publication charts directory
        os.makedirs(results_dir / "publication_charts", exist_ok=True)
        
        print("Generating publication-quality charts for paper...")
        
        # 1. Precision-recall chart
        create_precision_recall_chart(
            evaluator.results,
            output_path=str(results_dir / "publication_charts" / "precision_recall_tradeoff.png")
        )
        
        # 2. F1 vs time tradeoff chart (uses combined raw counts by default)
        create_tradeoff_chart(
            evaluator.results,
            output_path=str(results_dir / "publication_charts" / "f1_time_tradeoff.png")
        )
        
        # 3. Dataset comparison chart
        create_dataset_comparison_chart(
            evaluator.results,
            output_path=str(results_dir / "publication_charts" / "dataset_comparison.png")
        )
        
        # 4. Performance heatmaps for different metrics
        for metric in ['f1', 'precision', 'recall', 'time_per_char_seconds']:
            create_performance_heatmap(
                evaluator.results,
                output_path=str(results_dir / "publication_charts" / f"{metric}_heatmap.png"),
                metric=metric
            )
            
        print("Publication-quality charts generated successfully")
        
    except ImportError as e:
        print(f"Warning: Could not generate publication-quality charts. Error: {e}")
        print("You can run them separately using: python -m lsp.examples.publication_charts")
    
    # Now generate the HTML report AFTER all charts are created
    try:
        print("\n=== Step 6: Generate HTML Report with Charts ===")
        print("Generating HTML report with detailed sentence analysis and charts...")
        
        # First try using our enhanced HTML generator
        try:
            from lsp.regenerate_html import generate_custom_html_report
            html_output_path = str(results_dir / "evaluation_report.html")
            generate_custom_html_report(evaluator.results, html_output_path, max_examples_per_dataset=10)
            print(f"Enhanced HTML report saved to: {html_output_path}")
        except ImportError:
            # Fall back to the standard HTML generator
            print("Enhanced HTML generator not available, using standard generator...")
            html_output_path = str(results_dir / "evaluation_report.html")
            evaluator.generate_html_report(html_output_path, max_examples_per_dataset=10)
            print(f"HTML report saved to: {html_output_path}")
    except Exception as e:
        print(f"Error generating HTML report: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate a README for the results directory
    readme = f"""# Legal Sentence Boundary Detection Results

Results generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents

- `evaluation_results.json`: Complete evaluation results
- `evaluation_report.html`: Interactive HTML report with color-coded sentence analysis
- `dataset_stats.json`: Statistics for all datasets
- `tokenizers.json`: Information about tokenizers used

### Directories

- `charts/`: Visualizations of tokenizer performance
- `stats/`: Detailed statistics for each dataset
- `latex/`: LaTeX tables for the paper
- `publication_charts/`: Publication-quality charts for ACL/EMNLP paper

## Summary

- Tokenizers evaluated: {len(tokenizers)}
- Datasets analyzed: {len(datasets)}
- Evaluation sample size: {args.eval_sample if args.eval_sample else "all examples"}

To regenerate these results, run:
```
python src/lsp/examples/paper_workflow.py --output {args.output} --sample {args.sample or "None"} --eval-sample {args.eval_sample or "None"}
```
"""
    
    with open(results_dir / "README.md", "w") as f:
        f.write(readme)
    
    print(f"\nWorkflow complete! Results saved to: {results_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run complete paper workflow")
    parser.add_argument("--output", type=str, default="results", help="Directory for output files")
    parser.add_argument("--sample", type=int, help="Number of examples to load from each dataset")
    parser.add_argument("--eval-sample", type=int, help="Number of examples to use for evaluation")
    args = parser.parse_args()
    
    run_workflow(args)


if __name__ == "__main__":
    main()