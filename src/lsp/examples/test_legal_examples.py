"""
Test tokenizers with challenging legal text examples.

This script tests all implemented tokenizers against a variety of
legal text examples that present common challenges for sentence boundary detection.
"""

import os
import sys
import json
import warnings
from typing import Dict, List, Any, Tuple
from tabulate import tabulate

# Filter out spaCy's lemmatizer warnings
warnings.filterwarnings("ignore", message=".*W108.*lemmatizer did not find POS annotation.*")

# Import tokenizers
from lsp.tokenizers.nupunkt import NupunktTokenizer
from lsp.tokenizers.nltk import NLTKTokenizer
from lsp.tokenizers.spacy import SpacyTokenizer
from lsp.tokenizers.pysbd import PySBDTokenizer
from lsp.tokenizers.charboundary import CharBoundaryTokenizer

# Define challenging legal text examples
LEGAL_EXAMPLES = [
    {
        "name": "Basic two sentences",
        "text": "This is a simple first sentence. This is the second sentence.",
        "expected_count": 2
    },
    {
        "name": "Legal citation",
        "text": "The Court has recognized this principle in Smith v. Jones, 123 U.S. 456 (1950). The same reasoning applies here.",
        "expected_count": 2
    },
    {
        "name": "Abbreviations",
        "text": "The plaintiff, Mr. Smith, filed suit against the defendant, Corp. Inc., for breach of contract. The court ruled in favor of the plaintiff.",
        "expected_count": 2
    },
    {
        "name": "Numbered list",
        "text": "The contract includes the following terms: 1. Payment schedule. 2. Delivery timeline. 3. Warranty period.",
        "expected_count": 4  # Might vary by tokenizer
    },
    {
        "name": "Quotations",
        "text": "The statute states that \"all persons have the right to due process.\" This requirement applies in all cases.",
        "expected_count": 2
    },
    {
        "name": "Complex citation",
        "text": "In Brown v. Board of Education, 347 U.S. 483, 495 (1954), the Court held that separate educational facilities are inherently unequal. This ruling overturned Plessy v. Ferguson.",
        "expected_count": 2
    },
    {
        "name": "Multiple periods",
        "text": "The defendant violated Section 4.3.1 of the contract. The plaintiff is entitled to damages.",
        "expected_count": 2
    },
    {
        "name": "Parenthetical text",
        "text": "The evidence (including the video recording from January 20, 2023) shows clear liability. The defendant has failed to rebut this evidence.",
        "expected_count": 2
    },
    {
        "name": "Semi-colon",
        "text": "The contract was signed on January 1, 2023; however, the work did not begin until March. This delay constitutes a breach.",
        "expected_count": 2
    },
    {
        "name": "Bullet points",
        "text": "The agreement includes: • First term. • Second term. • Third term.",
        "expected_count": 4  # Might vary by tokenizer
    }
]


def initialize_tokenizers() -> List[Any]:
    """Initialize all tokenizers.
    
    Returns:
        List of initialized tokenizer instances
    """
    # Initialize nupunkt
    nupunkt = NupunktTokenizer()
    nupunkt.initialize()
    
    # Initialize NLTK
    nltk = NLTKTokenizer()
    nltk.initialize()
    
    # Initialize spaCy - small model
    spacy_sm = SpacyTokenizer(name="spacy_sm")
    spacy_sm.initialize(model="en_core_web_sm")
    
    # Initialize spaCy - large model (optional)
    try:
        spacy_lg = SpacyTokenizer(name="spacy_lg")
        spacy_lg.initialize(model="en_core_web_lg")
    except Exception as e:
        print(f"Note: spaCy large model could not be initialized: {e}")
        spacy_lg = None
    
    # Initialize PySBD
    pysbd = PySBDTokenizer()
    pysbd.initialize()
    
    # Initialize CharBoundary models
    cb_small = CharBoundaryTokenizer(name="charboundary_small")
    cb_small.initialize(size="small")
    
    cb_medium = CharBoundaryTokenizer(name="charboundary_medium")
    cb_medium.initialize(size="medium")
    
    cb_large = CharBoundaryTokenizer(name="charboundary_large")
    cb_large.initialize(size="large")
    
    # Return list of initialized tokenizers
    tokenizers = [nupunkt, nltk, spacy_sm, pysbd, cb_small, cb_medium, cb_large]
    if spacy_lg:
        tokenizers.insert(3, spacy_lg)  # Insert after spacy_sm
    
    return tokenizers


def test_example(tokenizers: List[Any], example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Test a single example with all tokenizers.
    
    Args:
        tokenizers: List of initialized tokenizers
        example: Example dictionary with name, text, and expected_count
        
    Returns:
        List of result dictionaries for each tokenizer
    """
    results = []
    
    for tokenizer in tokenizers:
        # Tokenize the text
        sentences = tokenizer.tokenize(example["text"])
        
        # Record the results
        result = {
            "tokenizer": tokenizer.name,
            "example": example["name"],
            "expected_count": example["expected_count"],
            "actual_count": len(sentences),
            "matches_expected": len(sentences) == example["expected_count"],
            "sentences": sentences
        }
        
        results.append(result)
        
    return results


def main():
    """Main function to run the example tests."""
    print("Initializing tokenizers...")
    tokenizers = initialize_tokenizers()
    print(f"Initialized {len(tokenizers)} tokenizers")
    
    # Run each example with all tokenizers
    all_results = []
    
    for example in LEGAL_EXAMPLES:
        print(f"\nTesting example: {example['name']}")
        results = test_example(tokenizers, example)
        all_results.extend(results)
        
        # Create a table for this example
        table_data = []
        for result in results:
            status = "✓" if result["matches_expected"] else "✗"
            table_data.append([
                result["tokenizer"],
                result["actual_count"],
                result["expected_count"],
                status
            ])
        
        headers = ["Tokenizer", "Actual Count", "Expected Count", "Match"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Print sentences for each tokenizer
        if any(not r["matches_expected"] for r in results):
            print("\nSentence boundaries by tokenizer:")
            for result in results:
                print(f"\n{result['tokenizer']}:")
                for i, sent in enumerate(result["sentences"]):
                    print(f"  {i+1}. {sent}")
    
    # Calculate overall performance
    tokenizer_stats = {}
    for result in all_results:
        tokenizer = result["tokenizer"]
        if tokenizer not in tokenizer_stats:
            tokenizer_stats[tokenizer] = {
                "total": 0,
                "correct": 0
            }
        
        tokenizer_stats[tokenizer]["total"] += 1
        if result["matches_expected"]:
            tokenizer_stats[tokenizer]["correct"] += 1
    
    # Print summary
    print("\n\nOverall Performance:")
    summary_data = []
    for tokenizer, stats in tokenizer_stats.items():
        accuracy = (stats["correct"] / stats["total"]) * 100
        summary_data.append([
            tokenizer,
            stats["correct"],
            stats["total"],
            f"{accuracy:.1f}%"
        ])
    
    headers = ["Tokenizer", "Correct", "Total", "Accuracy"]
    print(tabulate(summary_data, headers=headers, tablefmt="grid"))
    
    # Save results to file
    output_path = os.path.join(os.path.dirname(__file__), "tokenizer_test_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "examples": LEGAL_EXAMPLES,
            "results": all_results,
            "summary": {k: v for k, v in tokenizer_stats.items()}
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()