"""
Evaluate tokenizers against standardized legal examples.

This script evaluates all tokenizer implementations against
standardized legal examples with known expected sentence boundaries.
It reports detailed metrics on tokenizer performance.
"""

import os
import sys
import time
import json
import warnings
from typing import Dict, List, Any, Tuple, Optional
from tabulate import tabulate
import difflib

# Filter out spaCy's lemmatizer warnings
warnings.filterwarnings("ignore", message=".*W108.*lemmatizer did not find POS annotation.*")

# Import tokenizers
from lsp.core.tokenizer import SentenceTokenizer
from lsp.tokenizers.nupunkt import NupunktTokenizer
from lsp.tokenizers.nltk import NLTKTokenizer
from lsp.tokenizers.spacy import SpacyTokenizer
from lsp.tokenizers.pysbd import PySBDTokenizer
from lsp.tokenizers.charboundary import CharBoundaryTokenizer

# Import legal examples
from lsp.examples.legal_sentences import LEGAL_SENTENCES, get_all_examples


def initialize_tokenizers() -> Dict[str, SentenceTokenizer]:
    """Initialize all tokenizers.
    
    Returns:
        Dictionary of tokenizer name to initialized tokenizer instance
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


def evaluate_exact_match(predicted: List[str], expected: List[str]) -> Dict[str, Any]:
    """Evaluate exact match between predicted and expected sentences.
    
    Args:
        predicted: List of predicted sentences
        expected: List of expected sentences
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Count matches
    correct = len(predicted) == len(expected) and all(p == e for p, e in zip(predicted, expected))
    
    # Calculate metrics
    metrics = {
        "exact_match": correct,
        "predicted_count": len(predicted),
        "expected_count": len(expected),
        "count_match": len(predicted) == len(expected)
    }
    
    return metrics


def evaluate_fuzzy_match(predicted: List[str], expected: List[str], threshold: float = 0.9) -> Dict[str, Any]:
    """Evaluate fuzzy match between predicted and expected sentences.
    
    Args:
        predicted: List of predicted sentences
        expected: List of expected sentences
        threshold: Similarity threshold for considering a match (0.0-1.0)
        
    Returns:
        Dictionary with fuzzy evaluation metrics
    """
    # If empty lists, handle specially
    if not predicted and not expected:
        return {
            "matches": 0,
            "total": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "similarity_scores": []
        }
    elif not predicted:
        return {
            "matches": 0,
            "total": len(expected),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "similarity_scores": []
        }
    elif not expected:
        return {
            "matches": 0,
            "total": len(predicted),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "similarity_scores": []
        }
    
    # Calculate similarity matrix
    similarity_matrix = []
    for exp_sent in expected:
        row = []
        for pred_sent in predicted:
            similarity = difflib.SequenceMatcher(None, exp_sent, pred_sent).ratio()
            row.append(similarity)
        similarity_matrix.append(row)
    
    # Calculate precision and recall using threshold
    matches = 0
    matched_expected = set()
    matched_predicted = set()
    
    # Find matches above threshold
    for i, exp_sent in enumerate(expected):
        for j, pred_sent in enumerate(predicted):
            if similarity_matrix[i][j] >= threshold and j not in matched_predicted:
                matches += 1
                matched_expected.add(i)
                matched_predicted.add(j)
                break
    
    # Calculate metrics
    precision = matches / len(predicted) if predicted else 0
    recall = matches / len(expected) if expected else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Flatten similarity scores for reporting
    similarity_scores = [sim for row in similarity_matrix for sim in row]
    
    return {
        "matches": matches,
        "total": max(len(predicted), len(expected)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "similarity_scores": similarity_scores
    }


def evaluate_example(tokenizers: Dict[str, SentenceTokenizer], example: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate all tokenizers on a single example.
    
    Args:
        tokenizers: Dictionary of tokenizer name to tokenizer instance
        example: Example dictionary with text and expected_sentences
        
    Returns:
        Dictionary with evaluation results for each tokenizer
    """
    results = {}
    
    for name, tokenizer in tokenizers.items():
        # Tokenize the text
        start_time = time.time()
        sentences = tokenizer.tokenize(example["text"])
        end_time = time.time()
        tokenize_time = end_time - start_time
        
        # Evaluate exact match
        exact_metrics = evaluate_exact_match(sentences, example["expected_sentences"])
        
        # Evaluate fuzzy match
        fuzzy_metrics = evaluate_fuzzy_match(sentences, example["expected_sentences"])
        
        # Store results
        results[name] = {
            "tokenizer": name,
            "sentences": sentences,
            "tokenize_time": tokenize_time,
            "exact": exact_metrics,
            "fuzzy": fuzzy_metrics
        }
    
    return results


def print_example_results(example: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Print results for a single example.
    
    Args:
        example: Example dictionary
        results: Results dictionary for all tokenizers
    """
    print(f"\n=== Example: {example['name']} ===")
    print(f"Text: {example['text']}")
    print("\nExpected sentences:")
    for i, sent in enumerate(example["expected_sentences"]):
        print(f"  {i+1}. {sent}")
    
    # Prepare table data
    table_data = []
    for name, result in results.items():
        exact_match = "✓" if result["exact"]["exact_match"] else "✗"
        count_match = "✓" if result["exact"]["count_match"] else "✗"
        
        table_data.append([
            name,
            len(result["sentences"]),
            len(example["expected_sentences"]),
            count_match,
            exact_match,
            f"{result['fuzzy']['f1']:.3f}",
            f"{result['tokenize_time'] * 1000:.1f}"
        ])
    
    # Sort by F1 score descending
    table_data.sort(key=lambda x: float(x[5]), reverse=True)
    
    # Print table
    headers = ["Tokenizer", "Pred #", "Exp #", "Count Match", "Exact Match", "F1 Score", "Time (ms)"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print actual sentences for tokenizers that don't match
    for name, result in results.items():
        if not result["exact"]["exact_match"]:
            print(f"\n{name} sentences:")
            for i, sent in enumerate(result["sentences"]):
                print(f"  {i+1}. {sent}")


def calculate_overall_metrics(all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Calculate overall metrics across all examples.
    
    Args:
        all_results: Dictionary mapping tokenizer name to list of results
        
    Returns:
        Dictionary with overall metrics for each tokenizer
    """
    metrics = {}
    
    for tokenizer_name, results in all_results.items():
        # Initialize counters
        total_examples = len(results)
        exact_matches = 0
        count_matches = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_time = 0
        
        # Sum metrics across examples
        for result in results:
            if result["exact"]["exact_match"]:
                exact_matches += 1
            if result["exact"]["count_match"]:
                count_matches += 1
                
            total_precision += result["fuzzy"]["precision"]
            total_recall += result["fuzzy"]["recall"]
            total_f1 += result["fuzzy"]["f1"]
            total_time += result["tokenize_time"]
        
        # Calculate averages
        avg_precision = total_precision / total_examples if total_examples > 0 else 0
        avg_recall = total_recall / total_examples if total_examples > 0 else 0
        avg_f1 = total_f1 / total_examples if total_examples > 0 else 0
        avg_time = total_time / total_examples if total_examples > 0 else 0
        
        # Store metrics
        metrics[tokenizer_name] = {
            "exact_match_rate": exact_matches / total_examples if total_examples > 0 else 0,
            "count_match_rate": count_matches / total_examples if total_examples > 0 else 0,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_time": avg_time,
            "total_examples": total_examples,
            "exact_matches": exact_matches,
            "count_matches": count_matches
        }
    
    return metrics


def print_overall_results(metrics: Dict[str, Dict[str, Any]]) -> None:
    """Print overall results for all tokenizers.
    
    Args:
        metrics: Dictionary with overall metrics for each tokenizer
    """
    print("\n=== Overall Results ===")
    
    # Prepare table data
    table_data = []
    for name, metric in metrics.items():
        table_data.append([
            name,
            f"{metric['exact_match_rate']:.2f}",
            f"{metric['count_match_rate']:.2f}",
            f"{metric['avg_precision']:.3f}",
            f"{metric['avg_recall']:.3f}",
            f"{metric['avg_f1']:.3f}",
            f"{metric['avg_time'] * 1000:.1f}",
            f"{metric['exact_matches']}/{metric['total_examples']}"
        ])
    
    # Sort by F1 score descending
    table_data.sort(key=lambda x: float(x[5]), reverse=True)
    
    # Print table
    headers = ["Tokenizer", "Exact Rate", "Count Rate", "Precision", "Recall", "F1 Score", "Avg Time (ms)", "Exact/Total"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))


def main():
    """Main function to run the evaluation."""
    # Initialize tokenizers
    tokenizers = initialize_tokenizers()
    
    if not tokenizers:
        print("Error: No tokenizers could be initialized.")
        return
    
    # Get all examples
    examples = get_all_examples()
    
    # Store results by tokenizer name
    all_results = {name: [] for name in tokenizers.keys()}
    
    # Evaluate each example
    for example in examples:
        print(f"\nEvaluating example: {example['name']}")
        results = evaluate_example(tokenizers, example)
        
        # Store results
        for name, result in results.items():
            all_results[name].append(result)
        
        # Print results for this example
        print_example_results(example, results)
    
    # Calculate and print overall metrics
    overall_metrics = calculate_overall_metrics(all_results)
    print_overall_results(overall_metrics)
    
    # Save detailed results to file
    output_path = os.path.join(os.path.dirname(__file__), "tokenizer_evaluation.json")
    with open(output_path, "w") as f:
        json.dump({
            "examples": examples,
            "results": {name: [dict(r) for r in results] for name, results in all_results.items()},
            "metrics": overall_metrics
        }, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()