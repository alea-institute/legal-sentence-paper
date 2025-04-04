"""
Profile memory usage of tokenizers.

This script measures the memory usage of different tokenizers during
initialization and tokenization to understand their memory footprint.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import time
from typing import Dict, List, Any, Optional
import gc

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import memory profiler
from memory_profiler import memory_usage

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


def profile_tokenizer_initialization(tokenizer_name, tokenizer_class, kwargs=None, special_params=None):
    """Profile memory usage during tokenizer initialization.
    
    Args:
        tokenizer_name: Name of the tokenizer
        tokenizer_class: Tokenizer class
        kwargs: Initialization keyword arguments
        special_params: Special parameters for initialize method
    
    Returns:
        Tuple of (tokenizer, memory_usage_mb)
    """
    if kwargs is None:
        kwargs = {}
    if special_params is None:
        special_params = {}
    
    # Clear memory before profiling
    gc.collect()
    
    def init_func():
        tokenizer = tokenizer_class(**kwargs)
        if special_params:
            tokenizer.initialize(**special_params)
        else:
            tokenizer.initialize()
        return tokenizer
    
    # Measure memory usage during initialization
    mem_usage, tokenizer = memory_usage(
        (init_func, [], {}),
        retval=True,
        interval=0.1,
        include_children=True
    )
    
    # Calculate memory usage
    baseline = mem_usage[0]  # Memory usage at start
    max_usage = max(mem_usage)  # Peak memory usage
    init_memory = max_usage - baseline
    
    print(f"  {tokenizer_name} initialization memory usage: {init_memory:.2f} MB")
    
    return tokenizer, init_memory


def profile_tokenizer_tokenization(tokenizer, text, iterations=3):
    """Profile memory usage during tokenization.
    
    Args:
        tokenizer: Initialized tokenizer
        text: Text to tokenize
        iterations: Number of iterations to run
    
    Returns:
        Memory usage in MB
    """
    # Clear memory before profiling
    gc.collect()
    
    def tokenize_func():
        # Run multiple iterations to get stable reading
        for _ in range(iterations):
            tokenizer.tokenize(text)
    
    # Measure memory usage during tokenization
    mem_usage = memory_usage(
        (tokenize_func, [], {}),
        interval=0.1,
        include_children=True
    )
    
    # Calculate memory usage
    baseline = mem_usage[0]  # Memory usage at start
    max_usage = max(mem_usage)  # Peak memory usage
    tokenize_memory = max_usage - baseline
    
    print(f"  Tokenization memory usage: {tokenize_memory:.2f} MB")
    
    return tokenize_memory


def profile_tokenizer_bulk(tokenizer, texts, iterations=3):
    """Profile memory usage during bulk tokenization.
    
    Args:
        tokenizer: Initialized tokenizer
        texts: List of texts to tokenize
        iterations: Number of iterations to run
    
    Returns:
        Memory usage in MB
    """
    # Clear memory before profiling
    gc.collect()
    
    def tokenize_bulk_func():
        # Run multiple iterations to get stable reading
        for _ in range(iterations):
            for text in texts:
                tokenizer.tokenize(text)
    
    # Measure memory usage during tokenization
    mem_usage = memory_usage(
        (tokenize_bulk_func, [], {}),
        interval=0.1,
        include_children=True
    )
    
    # Calculate memory usage
    baseline = mem_usage[0]  # Memory usage at start
    max_usage = max(mem_usage)  # Peak memory usage
    bulk_memory = max_usage - baseline
    
    print(f"  Bulk tokenization memory usage: {bulk_memory:.2f} MB")
    
    return bulk_memory


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Profile memory usage of tokenizers")
    parser.add_argument("--tokenizers", type=str, nargs="+", help="Tokenizers to profile")
    parser.add_argument("--datasets", type=str, nargs="+", help="Datasets to use for profiling")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of examples from each dataset")
    parser.add_argument("--output", type=str, default="results/memory_profiles.json", help="Output file path")
    args = parser.parse_args()
    
    # Define tokenizer configurations
    available_tokenizers = {
        "nupunkt": (NupunktTokenizer, {}, {}),
        "nltk_punkt": (NLTKTokenizer, {}, {}),
        "spacy_sm": (SpacyTokenizer, {"name": "spacy_sm"}, {"model": "en_core_web_sm"}),
        "spacy_lg": (SpacyTokenizer, {"name": "spacy_lg"}, {"model": "en_core_web_lg"}),
        "pysbd": (PySBDTokenizer, {}, {}),
        "charboundary_small": (CharBoundaryTokenizer, {"name": "charboundary_small"}, {"size": "small"}),
        "charboundary_medium": (CharBoundaryTokenizer, {"name": "charboundary_medium"}, {"size": "medium"}),
        "charboundary_large": (CharBoundaryTokenizer, {"name": "charboundary_large"}, {"size": "large"})
    }
    
    # Filter tokenizers if requested
    if args.tokenizers:
        tokenizers_to_profile = {k: v for k, v in available_tokenizers.items() if k in args.tokenizers}
    else:
        tokenizers_to_profile = available_tokenizers
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = load_all_datasets(limit=args.limit)
    
    # Filter datasets if requested
    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}
    
    # Prepare results container
    results = {
        "timestamp": time.time(),
        "configuration": {
            "dataset_limit": args.limit
        },
        "results": {}
    }
    
    # Prepare sample texts for tokenization profiling
    # Use a fixed sample to ensure consistent measurements
    sample_texts = []
    for dataset_name, dataset in datasets.items():
        for i in range(min(args.limit, len(dataset))):
            example = dataset.get_example(i)
            text = dataset.get_text(example)
            sample_texts.append(text)
    
    # Create a concatenated sample for single-text profiling
    concatenated_sample = "\n\n".join(sample_texts[:3])  # Use first 3 texts
    
    # Profile each tokenizer
    for tokenizer_name, (tokenizer_class, init_kwargs, special_params) in tokenizers_to_profile.items():
        print(f"\n=== Profiling {tokenizer_name} ===")
        
        try:
            # Profile initialization
            tokenizer, init_memory = profile_tokenizer_initialization(
                tokenizer_name,
                tokenizer_class,
                init_kwargs,
                special_params
            )
            
            # Profile single text tokenization
            tokenize_memory = profile_tokenizer_tokenization(tokenizer, concatenated_sample)
            
            # Profile bulk tokenization
            bulk_memory = profile_tokenizer_bulk(tokenizer, sample_texts[:10])  # Use first 10 texts
            
            # Store results
            results["results"][tokenizer_name] = {
                "initialization_memory_mb": init_memory,
                "tokenization_memory_mb": tokenize_memory,
                "bulk_tokenization_memory_mb": bulk_memory,
                "total_memory_mb": init_memory + bulk_memory  # Total memory footprint
            }
            
        except Exception as e:
            print(f"  Error profiling {tokenizer_name}: {e}")
            results["results"][tokenizer_name] = {"error": str(e)}
    
    # Save results to file
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print a summary table
    print("\n=== Memory Usage Summary (MB) ===")
    print("Tokenizer\tInit\tTokenize\tBulk\tTotal")
    
    for tokenizer_name in results["results"]:
        if "error" not in results["results"][tokenizer_name]:
            res = results["results"][tokenizer_name]
            print(f"{tokenizer_name}\t{res['initialization_memory_mb']:.2f}\t{res['tokenization_memory_mb']:.2f}\t{res['bulk_tokenization_memory_mb']:.2f}\t{res['total_memory_mb']:.2f}")
        else:
            print(f"{tokenizer_name}\tERROR: {results['results'][tokenizer_name]['error']}")


if __name__ == "__main__":
    main()