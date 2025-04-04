#!/usr/bin/env python
"""Command-line interface for Legal Sentence Processing tokenizers."""

import sys
import argparse
import textwrap
import warnings
from typing import Dict, List, Any, Optional
import json

# Filter out common warnings to keep the output clean
warnings.filterwarnings("ignore", message=".*W108.*lemmatizer did not find POS annotation.*")

from lsp.tokenizers import (
    NupunktTokenizer,
    NLTKTokenizer,
    SpacyTokenizer,
    PySBDTokenizer,
    CharBoundaryTokenizer
)


def initialize_all_tokenizers():
    """Initialize all available tokenizers.
    
    Returns:
        Dictionary mapping tokenizer names to initialized tokenizer instances
    """
    tokenizers = {
        "nupunkt": NupunktTokenizer(),
        "nltk": NLTKTokenizer(),
        "spacy_sm": SpacyTokenizer(name="spacy_sm"),
        "spacy_lg": SpacyTokenizer(name="spacy_lg"),
        "pysbd": PySBDTokenizer(),
        "charboundary_small": CharBoundaryTokenizer(name="charboundary_small"),
        "charboundary_medium": CharBoundaryTokenizer(name="charboundary_medium"),
        "charboundary_large": CharBoundaryTokenizer(name="charboundary_large")
    }
    
    # Initialize with appropriate parameters
    tokenizers["nupunkt"].initialize()
    tokenizers["nltk"].initialize()
    tokenizers["spacy_sm"].initialize(model="en_core_web_sm")
    tokenizers["spacy_lg"].initialize(model="en_core_web_lg")
    tokenizers["pysbd"].initialize()
    tokenizers["charboundary_small"].initialize(size="small")
    tokenizers["charboundary_medium"].initialize(size="medium")
    tokenizers["charboundary_large"].initialize(size="large")
    
    # Perform warmup on all tokenizers
    for name, tokenizer in tokenizers.items():
        tokenizer.warmup()
        
    return tokenizers


def get_specific_tokenizers(tokenizer_names, all_tokenizers):
    """Get specific tokenizers by name.
    
    Args:
        tokenizer_names: List of tokenizer names to retrieve
        all_tokenizers: Dictionary of all available tokenizers
        
    Returns:
        Dictionary of requested tokenizers
    """
    result = {}
    for name in tokenizer_names:
        if name in all_tokenizers:
            result[name] = all_tokenizers[name]
        else:
            print(f"Warning: Tokenizer '{name}' not found. Available tokenizers: {', '.join(all_tokenizers.keys())}")
    return result


def format_text_output(text, tokenizer_results, max_width=100):
    """Format tokenization results as text.
    
    Args:
        text: Original input text
        tokenizer_results: Dictionary mapping tokenizer names to tokenization results
        max_width: Maximum width for text wrapping
        
    Returns:
        Formatted text output
    """
    output = []
    
    # Add a header with the original text
    output.append("Original text:")
    output.append("=" * 80)
    output.append(textwrap.fill(text, width=max_width))
    output.append("=" * 80)
    output.append("")
    
    # Add results for each tokenizer
    for tokenizer_name, sentences in sorted(tokenizer_results.items()):
        output.append(f"{tokenizer_name} ({len(sentences)} sentences):")
        output.append("-" * 80)
        
        for i, sentence in enumerate(sentences, 1):
            # Wrap the sentence text for better display
            wrapped = textwrap.fill(
                sentence, 
                width=max_width, 
                initial_indent=f"{i:2d}. ",
                subsequent_indent="    "
            )
            output.append(wrapped)
        
        output.append("")
    
    return "\n".join(output)


def format_json_output(text, tokenizer_results):
    """Format tokenization results as JSON.
    
    Args:
        text: Original input text
        tokenizer_results: Dictionary mapping tokenizer names to tokenization results
        
    Returns:
        JSON-formatted string
    """
    data = {
        "original_text": text,
        "results": {}
    }
    
    for tokenizer_name, sentences in tokenizer_results.items():
        data["results"][tokenizer_name] = {
            "count": len(sentences),
            "sentences": sentences
        }
    
    return json.dumps(data, indent=2)


def format_csv_output(text, tokenizer_results):
    """Format tokenization results as CSV.
    
    Args:
        text: Original input text
        tokenizer_results: Dictionary mapping tokenizer names to tokenization results
        
    Returns:
        CSV-formatted string
    """
    output = []
    
    # Add header row
    tokenizer_names = sorted(tokenizer_results.keys())
    output.append("sentence_num," + ",".join(tokenizer_names))
    
    # Find the maximum number of sentences across all tokenizers
    max_sentences = max(len(sentences) for sentences in tokenizer_results.values())
    
    # Add each sentence row
    for i in range(max_sentences):
        row = [str(i+1)]
        for name in tokenizer_names:
            sentences = tokenizer_results[name]
            cell = sentences[i] if i < len(sentences) else ""
            # Escape quotes and commas in CSV
            cell = cell.replace('"', '""')
            if "," in cell or "\n" in cell:
                cell = f'"{cell}"'
            row.append(cell)
        output.append(",".join(row))
    
    return "\n".join(output)


def tokenize_text(text, tokenizers):
    """Tokenize text using specified tokenizers.
    
    Args:
        text: Text to tokenize
        tokenizers: Dictionary of tokenizers to use
        
    Returns:
        Dictionary mapping tokenizer names to lists of sentences
    """
    results = {}
    
    for name, tokenizer in tokenizers.items():
        sentences = tokenizer.tokenize(text)
        results[name] = sentences
    
    return results


def main():
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Tokenize text using multiple sentence boundary detection algorithms"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", 
        type=str, 
        help="Text to tokenize"
    )
    input_group.add_argument(
        "--file", 
        type=str, 
        help="Path to file containing text to tokenize"
    )
    input_group.add_argument(
        "--stdin", 
        action="store_true", 
        help="Read text from standard input"
    )
    
    # Tokenizer selection
    parser.add_argument(
        "--tokenizers", 
        type=str, 
        nargs="+",
        help="Specific tokenizers to use (default: all available tokenizers)"
    )
    
    # Output format
    parser.add_argument(
        "--format", 
        type=str, 
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            return 1
    elif args.stdin:
        text = sys.stdin.read()
    
    # Initialize tokenizers
    all_tokenizers = initialize_all_tokenizers()
    
    # Select tokenizers to use
    if args.tokenizers:
        tokenizers = get_specific_tokenizers(args.tokenizers, all_tokenizers)
        if not tokenizers:
            print("No valid tokenizers specified", file=sys.stderr)
            return 1
    else:
        tokenizers = all_tokenizers
    
    # Tokenize the text
    results = tokenize_text(text, tokenizers)
    
    # Format and print the output
    if args.format == "text":
        output = format_text_output(text, results)
    elif args.format == "json":
        output = format_json_output(text, results)
    elif args.format == "csv":
        output = format_csv_output(text, results)
    
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())