"""
Benchmark throughput of tokenizers using batch processing.

This script evaluates tokenizer throughput using a bulk/batch approach 
that more accurately reflects the true speed of the underlying libraries.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import required modules
from lsp.dataloaders.discovery import load_all_datasets
from lsp.tokenizers.nupunkt import NupunktTokenizer
from lsp.tokenizers.nltk import NLTKTokenizer
from lsp.tokenizers.spacy import SpacyTokenizer
from lsp.tokenizers.pysbd import PySBDTokenizer
from lsp.tokenizers.charboundary import CharBoundaryTokenizer


def initialize_tokenizers(selected: Optional[List[str]] = None) -> Dict[str, Any]:
    """Initialize all tokenizers or a specific subset.
    
    Args:
        selected: Optional list of tokenizer names to initialize.
                If None, initialize all available tokenizers.
                
    Returns:
        Dictionary of tokenizer name to initialized tokenizer
    """
    available_tokenizers = {
        "nupunkt": (NupunktTokenizer, {}),
        "nltk_punkt": (NLTKTokenizer, {}),
        "spacy_sm": (SpacyTokenizer, {"name": "spacy_sm", "model": "en_core_web_sm"}),
        "spacy_lg": (SpacyTokenizer, {"name": "spacy_lg", "model": "en_core_web_lg"}),
        "pysbd": (PySBDTokenizer, {}),
        "charboundary_small": (CharBoundaryTokenizer, {"name": "charboundary_small"}),
        "charboundary_medium": (CharBoundaryTokenizer, {"name": "charboundary_medium"}),
        "charboundary_large": (CharBoundaryTokenizer, {"name": "charboundary_large"})
    }
    
    # Initialize parameters for charboundary sizes
    charboundary_params = {
        "charboundary_small": {"size": "small"},
        "charboundary_medium": {"size": "medium"},
        "charboundary_large": {"size": "large"}
    }
    
    # Filter by selected tokenizers if provided
    if selected:
        tokenizers_to_init = {k: v for k, v in available_tokenizers.items() if k in selected}
    else:
        tokenizers_to_init = available_tokenizers
    
    # Initialize tokenizers
    tokenizers = {}
    
    for name, (tokenizer_class, kwargs) in tokenizers_to_init.items():
        try:
            print(f"Initializing {name}...")
            tokenizer = tokenizer_class(**kwargs)
            
            # Handle special initialization for charboundary models
            if name.startswith("charboundary_") and name in charboundary_params:
                tokenizer.initialize(**charboundary_params[name])
            else:
                tokenizer.initialize()
                
            tokenizers[name] = tokenizer
            print(f"  Successfully initialized {name}, initialization time: {tokenizer.initialization_time:.4f}s")
        except Exception as e:
            print(f"  Failed to initialize {name}: {e}")
    
    return tokenizers


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark tokenizer throughput using batch processing")
    parser.add_argument("--tokenizers", type=str, nargs="+", help="Tokenizers to benchmark")
    parser.add_argument("--datasets", type=str, nargs="+", help="Datasets to use for benchmarking")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples from each dataset")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default="results/throughput_benchmarks.json", help="Output file path")
    args = parser.parse_args()
    
    # Initialize tokenizers
    tokenizers = initialize_tokenizers(args.tokenizers)
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = load_all_datasets(limit=args.limit)
    
    # Filter datasets if requested
    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}
    
    # Run benchmark
    results = {
        "timestamp": time.time(),
        "configuration": {
            "warmup_iterations": args.warmup,
            "benchmark_iterations": args.iterations,
            "dataset_limit": args.limit
        },
        "results": {}
    }
    
    # Create a benchmark for each tokenizer on each dataset
    for tokenizer_name, tokenizer in tokenizers.items():
        print(f"\n=== Benchmarking {tokenizer_name} ===")
        tokenizer_results = {}
        
        for dataset_name, dataset in datasets.items():
            print(f"\nDataset: {dataset_name} ({len(dataset)} examples)")
            
            try:
                # Run bulk evaluation
                result = dataset.evaluate_tokenizer_bulk(
                    tokenizer,
                    warmup_iterations=args.warmup,
                    iterations=args.iterations
                )
                
                # Store results
                tokenizer_results[dataset_name] = result
                
                # Print summary
                summary = result["summary"]
                print(f"  Precision: {summary['precision']:.4f}")
                print(f"  Recall: {summary['recall']:.4f}")
                print(f"  F1: {summary['f1']:.4f}")
                print(f"  Average throughput: {summary['throughput_chars_per_sec']:,.0f} chars/sec")
                print(f"  Maximum throughput: {summary['max_throughput_chars_per_sec']:,.0f} chars/sec")
                print(f"  Minimum processing time: {summary['min_time_seconds']:.4f}s")
                
            except Exception as e:
                print(f"  Error benchmarking {tokenizer_name} on {dataset_name}: {e}")
                tokenizer_results[dataset_name] = {"error": str(e)}
        
        # Store results for this tokenizer
        results["results"][tokenizer_name] = tokenizer_results
    
    # Save results to file
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Also print a summary table
    print("\n=== Throughput Summary (chars/sec) ===")
    print("Tokenizer\t" + "\t".join(datasets.keys()))
    
    for tokenizer_name in results["results"]:
        throughputs = []
        for dataset_name in datasets.keys():
            if dataset_name in results["results"][tokenizer_name]:
                res = results["results"][tokenizer_name][dataset_name]
                if "summary" in res and "max_throughput_chars_per_sec" in res["summary"]:
                    throughput = f"{res['summary']['max_throughput_chars_per_sec']:,.0f}"
                else:
                    throughput = "N/A"
            else:
                throughput = "N/A"
            throughputs.append(throughput)
        
        print(f"{tokenizer_name}\t" + "\t".join(throughputs))


if __name__ == "__main__":
    main()