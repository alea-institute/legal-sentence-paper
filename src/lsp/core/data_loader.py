"""Abstract base class for dataset loaders."""

import abc
import json
import os
from typing import Dict, List, Any, Tuple, Optional, Set, Iterator


class DataLoader(abc.ABC):
    """Abstract base class for dataset loaders.
    
    This interface defines a standard way to load and process datasets with
    different annotation formats for sentence boundary detection evaluation.
    Implementations should handle the specifics of each dataset format while
    conforming to this common interface.
    """
    
    def __init__(self, name: str):
        """Initialize the data loader.
        
        Args:
            name: Name identifier for the data loader
        """
        self.name = name
        self._dataset_path = None
        self._is_loaded = False
        self._data = []
        self._metadata = {}
        
    @property
    def is_loaded(self) -> bool:
        """Check if the dataset is loaded.
        
        Returns:
            True if the dataset is loaded, False otherwise
        """
        return self._is_loaded
    
    @property
    def dataset_path(self) -> Optional[str]:
        """Get the dataset path.
        
        Returns:
            Path to the dataset file or None if not set
        """
        return self._dataset_path
    
    @property
    def num_examples(self) -> int:
        """Get the number of examples in the dataset.
        
        Returns:
            Number of examples in the dataset
        """
        return len(self._data)
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get dataset metadata.
        
        Returns:
            Dictionary of metadata about the dataset
        """
        return self._metadata.copy()
    
    @abc.abstractmethod
    def load(self, dataset_path: str, **kwargs) -> None:
        """Load the dataset from a file.
        
        Args:
            dataset_path: Path to the dataset file
            **kwargs: Additional dataset-specific loading options
        """
        self._dataset_path = dataset_path
        self._is_loaded = False
        self._data = []
        self._metadata = {}
    
    def get_example(self, index: int) -> Dict[str, Any]:
        """Get a specific example from the dataset.
        
        Args:
            index: Index of the example to retrieve
            
        Returns:
            Dictionary with the example data
            
        Raises:
            IndexError: If the index is out of range
            RuntimeError: If the dataset is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("Dataset must be loaded before accessing examples.")
        
        if index < 0 or index >= len(self._data):
            raise IndexError(f"Index {index} out of range for dataset with {len(self._data)} examples.")
        
        return self._data[index]
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over examples in the dataset.
        
        Returns:
            Iterator of example dictionaries
            
        Raises:
            RuntimeError: If the dataset is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("Dataset must be loaded before iterating.")
        
        return iter(self._data)
    
    def __len__(self) -> int:
        """Get the number of examples in the dataset.
        
        Returns:
            Number of examples in the dataset
        """
        return len(self._data)
    
    @abc.abstractmethod
    def get_text(self, example: Dict[str, Any]) -> str:
        """Get the raw text from an example.
        
        Args:
            example: Example dictionary
            
        Returns:
            Raw text string
        """
        pass
    
    @abc.abstractmethod
    def get_sentences(self, example: Dict[str, Any]) -> List[str]:
        """Get the sentences from an example.
        
        Args:
            example: Example dictionary
            
        Returns:
            List of sentence strings
        """
        pass
    
    @abc.abstractmethod
    def get_spans(self, example: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Get the character spans for each sentence from an example.
        
        Args:
            example: Example dictionary
            
        Returns:
            List of (start, end) tuples representing sentence spans
        """
        pass
    
    def normalize_sentences(self, sentences: List[str]) -> List[str]:
        """Normalize sentences to handle punctuation and whitespace.
        
        Args:
            sentences: List of sentence strings
            
        Returns:
            List of normalized sentence strings
        """
        # Default implementation: trim whitespace
        return [s.strip() for s in sentences]
    
    def normalize_text(self, text: str) -> str:
        """Normalize text to handle punctuation and whitespace consistently.
        
        Args:
            text: Input text string
            
        Returns:
            Normalized text string
        """
        # Default implementation: preserve original text
        return text
    
    def evaluate_tokenizer(self, tokenizer, example_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """Evaluate a tokenizer on the dataset.
        
        Args:
            tokenizer: An initialized SentenceTokenizer instance
            example_indices: Optional list of example indices to evaluate on
                            If None, evaluates on all examples
                            
        Returns:
            Dictionary of evaluation results
            
        Raises:
            RuntimeError: If the dataset is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("Dataset must be loaded before evaluation.")
        
        if not tokenizer.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before evaluation.")
        
        # Determine which examples to evaluate
        if example_indices is None:
            examples = self._data
        else:
            examples = [self.get_example(i) for i in example_indices]
        
        results = {
            "tokenizer": tokenizer.name,
            "dataset": self.name,
            "num_examples": len(examples),
            "examples": []
        }
        
        # Totals for summary metrics
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        total_chars = 0
        total_sentences = 0
        total_time = 0
        
        # Create a detailed example store to keep all raw samples and annotations
        detailed_examples = []
        
        # Evaluate each example
        for example in examples:
            # Get ground truth
            text = self.get_text(example)
            true_sentences = self.get_sentences(example)
            true_spans = self.get_spans(example)
            
            # Normalize the text
            normalized_text = self.normalize_text(text)
            
            # Run tokenizer benchmark
            benchmark = tokenizer.benchmark(normalized_text, iterations=1)
            
            # Get tokenizer results
            pred_sentences = tokenizer.tokenize(normalized_text)
            pred_spans = tokenizer.get_spans(normalized_text)
            
            # Evaluate boundary detection accuracy
            tp, fp, fn = self._evaluate_boundaries(true_spans, pred_spans)
            
            # Update totals
            total_true_positives += tp
            total_false_positives += fp
            total_false_negatives += fn
            total_chars += len(text)
            total_sentences += len(true_sentences)
            total_time += benchmark["avg_time_seconds"]
            
            # Create a detailed example result with all information needed for visualization
            example_result = {
                "text": text,
                "true_sentences": true_sentences,
                "true_spans": true_spans,
                "pred_sentences": pred_sentences,
                "pred_spans": pred_spans,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "chars": len(text),
                "time_seconds": benchmark["avg_time_seconds"]
            }
            
            # Store all examples for later processing and visualization
            detailed_examples.append(example_result)
            
            # Add to results - always include at least the first 10 examples
            # for visualization purposes
            if len(results["examples"]) < 10 or len(text) < 10000:  # Include all small examples
                results["examples"].append(example_result)
        
        # Always store all detailed examples in the results
        # This ensures they're available for visualization and error analysis
        results["detailed_examples"] = detailed_examples
        
        # Calculate summary metrics
        precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
        recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results["summary"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": total_true_positives / (total_true_positives + total_false_positives + total_false_negatives) if (total_true_positives + total_false_positives + total_false_negatives) > 0 else 0,
            "total_true_positives": total_true_positives,
            "total_false_positives": total_false_positives,
            "total_false_negatives": total_false_negatives,
            "total_chars": total_chars,
            "total_sentences": total_sentences,
            "total_time_seconds": total_time,
            "time_per_char_seconds": total_time / total_chars if total_chars > 0 else 0,
            "time_per_sentence_seconds": total_time / total_sentences if total_sentences > 0 else 0
        }
        
        return results
        
    def evaluate_tokenizer_bulk(self, tokenizer, example_indices: Optional[List[int]] = None, 
                               warmup_iterations: int = 1, iterations: int = 3) -> Dict[str, Any]:
        """Evaluate a tokenizer on the dataset using bulk processing for accurate throughput measurement.
        
        This method provides a more accurate throughput measurement by processing
        all texts in a batch, similar to how the original libraries benchmark themselves.
        
        Args:
            tokenizer: An initialized SentenceTokenizer instance
            example_indices: Optional list of example indices to evaluate on
                            If None, evaluates on all examples
            warmup_iterations: Number of warmup iterations to perform
            iterations: Number of timed iterations to perform
                            
        Returns:
            Dictionary of evaluation results with bulk throughput measurements
            
        Raises:
            RuntimeError: If the dataset is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("Dataset must be loaded before evaluation.")
        
        if not tokenizer.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before evaluation.")
        
        # Determine which examples to evaluate
        if example_indices is None:
            examples = self._data
        else:
            examples = [self.get_example(i) for i in example_indices]
            
        # Extract and normalize all texts
        all_texts = []
        total_true_sentences = 0
        total_chars = 0
        
        for example in examples:
            text = self.get_text(example)
            normalized_text = self.normalize_text(text)
            all_texts.append(normalized_text)
            total_chars += len(text)
            total_true_sentences += len(self.get_sentences(example))
            
        print(f"Evaluating {tokenizer.name} on {len(all_texts)} texts with {total_chars:,} characters...")
        
        # Run bulk benchmark
        bulk_results = tokenizer.benchmark_bulk(
            all_texts, 
            iterations=iterations, 
            warmup_iterations=warmup_iterations
        )
        
        # Also run accuracy evaluation on a small sample for metrics
        # This avoids the overhead of computing accuracy on the full dataset
        accuracy_sample_size = min(len(examples), 10)  # Use at most 10 examples for accuracy evaluation
        accuracy_sample = examples[:accuracy_sample_size]
        
        # Calculate accuracy metrics on the sample
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for example in accuracy_sample:
            text = self.get_text(example)
            normalized_text = self.normalize_text(text)
            true_spans = self.get_spans(example)
            
            # Get tokenizer predictions
            pred_spans = tokenizer.get_spans(normalized_text)
            
            # Evaluate boundary detection accuracy
            tp, fp, fn = self._evaluate_boundaries(true_spans, pred_spans)
            
            # Update totals
            true_positives += tp
            false_positives += fp
            false_negatives += fn
        
        # Calculate accuracy metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Combine results
        results = {
            "tokenizer": tokenizer.name,
            "dataset": self.name,
            "num_examples": len(examples),
            "bulk_benchmark": bulk_results,
            "summary": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0,
                "total_true_positives": true_positives,
                "total_false_positives": false_positives,
                "total_false_negatives": false_negatives,
                "total_chars": total_chars,
                "total_sentences": total_true_sentences,
                "total_time_seconds": bulk_results["avg_time_seconds"],
                "min_time_seconds": bulk_results["min_time_seconds"],
                "throughput_chars_per_sec": bulk_results["throughput_chars_per_sec"],
                "max_throughput_chars_per_sec": bulk_results["max_throughput_chars_per_sec"],
                "time_per_char_seconds": bulk_results["time_per_char_seconds"],
                "time_per_sentence_seconds": bulk_results["time_per_sentence_seconds"]
            }
        }
        
        return results
    
    def _evaluate_boundaries(self, true_spans: List[Tuple[int, int]], pred_spans: List[Tuple[int, int]]) -> Tuple[int, int, int]:
        """Evaluate boundary detection accuracy.
        
        Args:
            true_spans: List of (start, end) tuples for true sentences
            pred_spans: List of (start, end) tuples for predicted sentences
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        # Convert spans to boundary indices
        true_boundaries = set()
        for start, end in true_spans:
            true_boundaries.add(start)
            true_boundaries.add(end)
        
        pred_boundaries = set()
        for start, end in pred_spans:
            pred_boundaries.add(start)
            pred_boundaries.add(end)
        
        # Calculate metrics
        true_positives = len(true_boundaries.intersection(pred_boundaries))
        false_positives = len(pred_boundaries - true_boundaries)
        false_negatives = len(true_boundaries - pred_boundaries)
        
        return true_positives, false_positives, false_negatives
    
    def calculate_statistics(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Calculate statistics about the dataset.
        
        Args:
            sample_size: Optional number of examples to sample for statistics
            
        Returns:
            Dictionary of statistics
        """
        if not self.is_loaded:
            raise RuntimeError("Dataset must be loaded before calculating statistics.")
        
        import random
        
        # Determine examples to analyze
        if sample_size and sample_size < len(self._data):
            indices = random.sample(range(len(self._data)), sample_size)
        else:
            indices = range(len(self._data))
        
        stats = {
            "total_examples": len(self._data),
            "analyzed_examples": len(indices),
            "total_sentences": 0,
            "total_chars": 0,
            "sentence_lengths": [],
            "sentences_per_example": []
        }
        
        # Analyze each example
        for idx in indices:
            example = self.get_example(idx)
            text = self.get_text(example)
            sentences = self.get_sentences(example)
            
            stats["total_chars"] += len(text)
            stats["total_sentences"] += len(sentences)
            stats["sentences_per_example"].append(len(sentences))
            
            for sentence in sentences:
                stats["sentence_lengths"].append(len(sentence))
        
        # Calculate derived statistics
        if stats["total_sentences"] > 0:
            stats["avg_sentence_length"] = sum(stats["sentence_lengths"]) / len(stats["sentence_lengths"])
            stats["avg_sentences_per_example"] = sum(stats["sentences_per_example"]) / len(stats["sentences_per_example"])
            stats["min_sentence_length"] = min(stats["sentence_lengths"]) if stats["sentence_lengths"] else 0
            stats["max_sentence_length"] = max(stats["sentence_lengths"]) if stats["sentence_lengths"] else 0
            
            # Calculate sentence length distribution
            bins = [0, 50, 100, 150, 200, 300, 500, 1000, float('inf')]
            bin_counts = [0] * (len(bins) - 1)
            
            for length in stats["sentence_lengths"]:
                for i in range(len(bins) - 1):
                    if bins[i] <= length < bins[i+1]:
                        bin_counts[i] += 1
                        break
            
            # Create distribution
            stats["sentence_length_distribution"] = {
                f"{bins[i]}-{bins[i+1]-1 if bins[i+1] != float('inf') else 'inf'}": bin_counts[i]
                for i in range(len(bins) - 1)
            }
        else:
            stats["avg_sentence_length"] = 0
            stats["avg_sentences_per_example"] = 0
            stats["min_sentence_length"] = 0
            stats["max_sentence_length"] = 0
            stats["sentence_length_distribution"] = {}
        
        return stats
    
    def __str__(self) -> str:
        """Get a string representation of the data loader.
        
        Returns:
            String representation of the data loader
        """
        status = "loaded" if self.is_loaded else "not loaded"
        examples = f"{self.num_examples} examples" if self.is_loaded else "no examples"
        return f"{self.name} ({status}, {examples})"