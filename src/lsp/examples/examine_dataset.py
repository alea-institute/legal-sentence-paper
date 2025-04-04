"""
Examine a specific dataset in detail.

This utility script loads a dataset and provides detailed information
about its contents, structure, and examples.
"""

import sys
import argparse
import json
import random
from pathlib import Path
from tabulate import tabulate

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lsp.dataloaders.discovery import discover_datasets, load_dataset, create_loader_for_path


def print_example(dataset, index, show_sentences=True, max_sentences=5):
    """Print a detailed example from the dataset.
    
    Args:
        dataset: Loaded dataset
        index: Example index to show
        show_sentences: Whether to show sentences
        max_sentences: Maximum number of sentences to show
    """
    example = dataset.get_example(index)
    text = dataset.get_text(example)
    sentences = dataset.get_sentences(example)
    spans = dataset.get_spans(example)
    
    print(f"\n=== Example {index+1} ===")
    
    # Show text with length
    print(f"Text ({len(text)} chars):")
    print(f"{text[:200]}{'...' if len(text) > 200 else ''}")
    
    # Show number of sentences
    print(f"\nSentences: {len(sentences)}")
    
    # Show sentences if requested
    if show_sentences:
        for i, sentence in enumerate(sentences[:max_sentences]):
            span = spans[i] if i < len(spans) else (0, 0)
            print(f"  {i+1}. [{span[0]}:{span[1]}] {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
        
        if len(sentences) > max_sentences:
            print(f"  ... ({len(sentences) - max_sentences} more sentences)")


def analyze_dataset(dataset, sample_size=None):
    """Analyze the dataset and print statistics.
    
    Args:
        dataset: Loaded dataset
        sample_size: Optional sample size for analysis
    """
    # Use the built-in statistics method
    stats = dataset.calculate_statistics(sample_size=sample_size)
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Analyzed examples: {stats['analyzed_examples']}")
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Total characters: {stats['total_chars']}")
    print(f"Average sentences per example: {stats['avg_sentences_per_example']:.2f}")
    print(f"Average sentence length: {stats['avg_sentence_length']:.2f} chars")
    print(f"Minimum sentence length: {stats['min_sentence_length']} chars")
    print(f"Maximum sentence length: {stats['max_sentence_length']} chars")
    
    # Print sentence length distribution
    print("\nSentence Length Distribution:")
    distribution = stats["sentence_length_distribution"]
    for range_str, count in sorted(distribution.items(), key=lambda x: int(x[0].split("-")[0])):
        percentage = (count / stats['total_sentences']) * 100 if stats['total_sentences'] > 0 else 0
        print(f"  {range_str} chars: {count} sentences ({percentage:.1f}%)")
    
    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Examine a specific dataset in detail")
    parser.add_argument("dataset", type=str, help="Dataset ID or file path")
    parser.add_argument("--example", type=int, help="Show a specific example")
    parser.add_argument("--random", type=int, default=3, help="Show N random examples (default: 3)")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset statistics")
    parser.add_argument("--sample", type=int, default=100, help="Sample size for analysis (default: 100)")
    parser.add_argument("--limit", type=int, help="Limit number of examples to load")
    parser.add_argument("--export", type=str, help="Export dataset info to JSON file")
    args = parser.parse_args()
    
    # Check if the argument is a file path
    if Path(args.dataset).exists():
        print(f"Loading dataset from file: {args.dataset}")
        dataset = create_loader_for_path(args.dataset, limit=args.limit)
        if not dataset:
            print(f"Error: Could not determine dataset type for {args.dataset}")
            return
    else:
        # Try to find dataset by ID
        datasets = discover_datasets()
        if args.dataset not in datasets:
            print(f"Error: Dataset '{args.dataset}' not found. Available datasets:")
            for dataset_id in datasets.keys():
                print(f"  - {dataset_id}")
            return
        
        # Load the dataset
        print(f"Loading dataset: {datasets[args.dataset]['name']}")
        dataset = load_dataset(datasets[args.dataset], limit=args.limit)
    
    print(f"Loaded {len(dataset)} examples")
    
    # Print dataset metadata
    print("\n=== Dataset Metadata ===")
    for key, value in dataset.metadata.items():
        print(f"{key}: {value}")
    
    # Show specific example if requested
    if args.example is not None:
        if args.example >= len(dataset):
            print(f"Error: Example index {args.example} out of range (0-{len(dataset)-1})")
        else:
            print_example(dataset, args.example)
    
    # Show random examples if requested
    if args.random > 0:
        print(f"\n=== {args.random} Random Examples ===")
        if args.random >= len(dataset):
            indices = list(range(len(dataset)))
        else:
            indices = random.sample(range(len(dataset)), args.random)
        
        for idx in indices:
            print_example(dataset, idx)
    
    # Analyze dataset if requested
    stats = None
    if args.analyze:
        stats = analyze_dataset(dataset, args.sample)
    
    # Export dataset info if requested
    if args.export:
        export_data = {
            "dataset": {
                "name": dataset.name,
                "path": dataset.dataset_path,
                "examples": len(dataset),
                "metadata": dataset.metadata
            }
        }
        
        if stats:
            export_data["statistics"] = stats
        
        # Export example details
        if args.example is not None:
            example = dataset.get_example(args.example)
            export_data["example"] = {
                "index": args.example,
                "text": dataset.get_text(example),
                "sentences": dataset.get_sentences(example),
                "spans": dataset.get_spans(example)
            }
        
        # Write to file
        with open(args.export, "w") as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\nExported dataset info to {args.export}")


if __name__ == "__main__":
    main()