"""MultiLegalSBD data loader."""

import json
import os
from typing import Dict, List, Any, Tuple, Optional

from lsp.core.data_loader import DataLoader


class MultiLegalSBDLoader(DataLoader):
    """Data loader for MultiLegalSBD dataset.
    
    This dataset uses span annotation format.
    Format: text with spans that mark sentence boundaries
    
    Example:
    {
        "text": "This is the first sentence. This is the second sentence.",
        "spans": [
            {"start": 0, "end": 29, "label": "Sentence"},
            {"start": 30, "end": 58, "label": "Sentence"}
        ]
    }
    """
    
    def __init__(self, name: str = "multilegal"):
        """Initialize the MultiLegalSBD data loader.
        
        Args:
            name: Name identifier for the data loader
        """
        super().__init__(name)
    
    def load(self, dataset_path: str, **kwargs) -> None:
        """Load the dataset from a JSONL file.
        
        Args:
            dataset_path: Path to the dataset file
            **kwargs: Additional loading options:
                - limit: Maximum number of examples to load (default: all)
                - skip_invalid: Whether to skip invalid examples (default: True)
                - label: Span label to filter for (default: "Sentence")
        """
        super().load(dataset_path, **kwargs)
        
        limit = kwargs.get("limit")
        skip_invalid = kwargs.get("skip_invalid", True)
        span_label = kwargs.get("label", "Sentence")
        
        self._data = []
        
        # Load the JSONL file
        with open(dataset_path, 'r', encoding='utf-8') as f:
            line_number = 0
            valid_examples = 0
            invalid_examples = 0
            
            for line in f:
                line_number += 1
                
                # Parse JSONL
                try:
                    example = json.loads(line.strip())
                except json.JSONDecodeError:
                    if not skip_invalid:
                        raise
                    invalid_examples += 1
                    continue
                
                # Check if the example has the required fields
                if "text" not in example or "spans" not in example:
                    if skip_invalid:
                        invalid_examples += 1
                        continue
                    else:
                        raise ValueError(f"Example at line {line_number} is missing required fields")
                
                # Add line number for reference and span label
                example["_line_number"] = line_number
                example["_span_label"] = span_label
                
                # Filter spans by label if provided
                if span_label:
                    example["spans"] = [span for span in example["spans"] if span.get("label") == span_label]
                
                # Add example to dataset
                self._data.append(example)
                valid_examples += 1
                
                # Stop if we've reached the limit
                if limit and valid_examples >= limit:
                    break
        
        self._is_loaded = True
        
        # Set metadata
        self._metadata = {
            "path": dataset_path,
            "filename": os.path.basename(dataset_path),
            "format": "MultiLegalSBD spans",
            "span_label": span_label,
            "valid_examples": valid_examples,
            "invalid_examples": invalid_examples,
            "total_lines": line_number
        }
    
    def get_text(self, example: Dict[str, Any]) -> str:
        """Get the raw text from an example.
        
        Args:
            example: Example dictionary
            
        Returns:
            Raw text string
        """
        if "text" not in example:
            raise ValueError("Example is missing 'text' field")
        
        return example["text"]
    
    def get_sentences(self, example: Dict[str, Any]) -> List[str]:
        """Get the sentences from an example.
        
        Args:
            example: Example dictionary
            
        Returns:
            List of sentence strings extracted from spans
        """
        if "text" not in example or "spans" not in example:
            raise ValueError("Example is missing required fields")
        
        text = example["text"]
        spans = example["spans"]
        
        # Sort spans by start position
        sorted_spans = sorted(spans, key=lambda x: x.get("start", 0))
        
        # Extract sentences from spans
        sentences = []
        for span in sorted_spans:
            start = span.get("start", 0)
            end = span.get("end", 0)
            
            # Ensure valid spans
            if start < 0 or end > len(text) or start >= end:
                continue
                
            sentence = text[start:end]
            sentences.append(sentence)
        
        # Normalize sentences
        return self.normalize_sentences(sentences)
    
    def get_spans(self, example: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Get the character spans for each sentence from an example.
        
        Args:
            example: Example dictionary
            
        Returns:
            List of (start, end) tuples representing sentence spans
        """
        if "spans" not in example:
            raise ValueError("Example is missing 'spans' field")
        
        spans = example["spans"]
        
        # Sort spans by start position
        sorted_spans = sorted(spans, key=lambda x: x.get("start", 0))
        
        # Convert to (start, end) tuples
        return [(span.get("start", 0), span.get("end", 0)) for span in sorted_spans]
    
    def normalize_sentences(self, sentences: List[str]) -> List[str]:
        """Normalize sentences to handle punctuation and whitespace.
        
        Args:
            sentences: List of sentence strings
            
        Returns:
            List of normalized sentence strings
        """
        # MultiLegalSBD specific normalization: trim whitespace
        normalized = []
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
            
            normalized.append(sentence.strip())
        
        return normalized