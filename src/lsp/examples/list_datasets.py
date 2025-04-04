"""
List all available datasets found in the project.

This utility script finds and displays information about all
available datasets for legal sentence boundary detection evaluation.
"""

import sys
import argparse
from pathlib import Path
from tabulate import tabulate

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lsp.dataloaders.discovery import discover_datasets, load_dataset


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="List available datasets")
    parser.add_argument("--preview", action="store_true", help="Preview sample from each dataset")
    parser.add_argument("--sample", type=int, default=1, help="Number of examples to preview (default: 1)")
    args = parser.parse_args()
    
    # Discover available datasets
    print("Discovering datasets...")
    datasets = discover_datasets()
    
    if not datasets:
        print("No datasets found. Please check that data/ directory exists.")
        return
    
    # Prepare table data
    table_data = []
    for dataset_id, info in datasets.items():
        table_data.append([
            dataset_id,
            info["name"],
            info["format"],
            info["path"]
        ])
    
    # Sort by dataset ID
    table_data.sort(key=lambda x: x[0])
    
    # Print table
    print("\nAvailable Datasets:")
    headers = ["ID", "Name", "Format", "Path"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Preview if requested
    if args.preview:
        print("\nPreviewing datasets:")
        
        for dataset_id, info in datasets.items():
            print(f"\n=== {info['name']} ===")
            
            try:
                # Load dataset with limit
                dataset = load_dataset(info, limit=args.sample)
                
                print(f"Loaded {len(dataset)} examples")
                
                # Print sample
                for i in range(min(args.sample, len(dataset))):
                    example = dataset.get_example(i)
                    text = dataset.get_text(example)
                    sentences = dataset.get_sentences(example)
                    
                    print(f"\nExample {i+1}:")
                    print(f"Text ({len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
                    print(f"Sentences ({len(sentences)}):")
                    
                    for j, sentence in enumerate(sentences[:3]):
                        print(f"  {j+1}. {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
                    
                    if len(sentences) > 3:
                        print(f"  ... ({len(sentences) - 3} more sentences)")
                
            except Exception as e:
                print(f"Error loading dataset: {e}")


if __name__ == "__main__":
    main()