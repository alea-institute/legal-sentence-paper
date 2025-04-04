"""Example script to compare all tokenizers."""

import time
import warnings
from typing import Dict, List
import pandas as pd
from tabulate import tabulate

# Filter out common warnings to keep the output clean
warnings.filterwarnings("ignore", message=".*W108.*lemmatizer did not find POS annotation.*")

from lsp.tokenizers import (
    NupunktTokenizer,
    NLTKTokenizer,
    SpacyTokenizer,
    PySBDTokenizer,
    CharBoundaryTokenizer
)


def initialize_tokenizers(perform_warmup=True):
    """Initialize all tokenizers and return them in a dictionary.
    
    Args:
        perform_warmup: Whether to perform a warmup tokenization for each tokenizer
    
    Returns:
        Dictionary mapping tokenizer names to initialized tokenizer instances
    """
    tokenizers = {
        # Initialize all tokenizers
        "nupunkt": NupunktTokenizer(),
        "nltk_punkt": NLTKTokenizer(),
        "spacy_sm": SpacyTokenizer(name="spacy_sm"),
        "spacy_lg": SpacyTokenizer(name="spacy_lg"),
        "pysbd": PySBDTokenizer(),
        "charboundary_small": CharBoundaryTokenizer(name="charboundary_small"),
        "charboundary_medium": CharBoundaryTokenizer(name="charboundary_medium"),
        "charboundary_large": CharBoundaryTokenizer(name="charboundary_large")
    }
    
    # Initialize each tokenizer with appropriate options and perform warmup
    print("Initializing tokenizers...")
    for name, tokenizer in tokenizers.items():
        # Initialize with appropriate parameters
        if name == "spacy_sm":
            tokenizer.initialize(model="en_core_web_sm")
        elif name == "spacy_lg":
            tokenizer.initialize(model="en_core_web_lg")
        elif name == "charboundary_small":
            tokenizer.initialize(size="small")
        elif name == "charboundary_medium":
            tokenizer.initialize(size="medium")
        elif name == "charboundary_large":
            tokenizer.initialize(size="large")
        else:
            tokenizer.initialize()
        
        # Perform warmup tokenization if requested
        if perform_warmup:
            try:
                sentences = tokenizer.warmup()
                print(f"  {name}: Warmup successful ({len(sentences)} sentences)")
            except Exception as e:
                print(f"  {name}: Warmup failed - {str(e)}")
    
    return tokenizers


def compare_tokenization(tokenizers, text):
    """Compare tokenization results from all tokenizers."""
    results = {}
    
    for name, tokenizer in tokenizers.items():
        start_time = time.time()
        sentences = tokenizer.tokenize(text)
        end_time = time.time()
        
        results[name] = {
            "sentences": len(sentences),
            "time": end_time - start_time,
            "first_sentence": sentences[0] if sentences else "",
            "last_sentence": sentences[-1] if sentences else ""
        }
    
    return results


def compare_benchmarks(tokenizers, text, iterations=3):
    """Run benchmarks on all tokenizers and compare results."""
    results = []
    
    for name, tokenizer in tokenizers.items():
        benchmark = tokenizer.benchmark(text, iterations=iterations)
        
        results.append({
            "Tokenizer": name,
            "Sentences": benchmark["sentences"],
            "Avg Time (s)": f"{benchmark['avg_time_seconds']:.4f}",
            "Time/Char (μs)": f"{benchmark['time_per_char_seconds'] * 1e6:.2f}",
            "Time/Sent (ms)": f"{benchmark['time_per_sentence_seconds'] * 1e3:.2f}",
            "Throughput (char/s)": f"{benchmark['throughput_chars_per_sec']:.0f}"
        })
    
    return pd.DataFrame(results)


def main():
    """Run a comparison of all tokenizers on sample legal text."""
    # Sample legal text with multiple sentences
    text = """The Court has recognized that the Fourth Amendment protects citizens against
government demands for information. See Smith v. Maryland, 442 U.S. 735, 745-46 (1979).
In this case, the district court denied the preliminary injunction, inter alia, because 
it had "some doubt as to the copyrightability of the programs." Apple Computer, Inc. v. 
Franklin Computer Corp., 545 F. Supp. 812, 215 USPQ 935 (E.D. Pa. 1982). 
This legal ruling is fundamental to all future proceedings in this action and, as the 
parties and amici curiae seem to agree, has considerable significance to the computer
services industry."""
    
    # Initialize all tokenizers
    print("Initializing tokenizers...")
    tokenizers = initialize_tokenizers()
    
    # Compare tokenization
    print("\nComparing tokenization results:")
    tokenization_results = compare_tokenization(tokenizers, text)
    
    # Print number of sentences detected by each tokenizer
    sentence_counts = {name: result["sentences"] for name, result in tokenization_results.items()}
    print("Sentence counts:", sentence_counts)
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    benchmark_df = compare_benchmarks(tokenizers, text, iterations=5)
    
    # Print benchmark results
    print("\nBenchmark Results:")
    print(tabulate(benchmark_df, headers="keys", tablefmt="psql", showindex=False))


if __name__ == "__main__":
    main()