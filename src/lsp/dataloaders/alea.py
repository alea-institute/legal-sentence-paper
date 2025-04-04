"""ALEA Legal Benchmark data loader."""

import json
import os
from typing import Dict, List, Any, Tuple, Optional

from lsp.core.data_loader import DataLoader


class AleaDataLoader(DataLoader):
    """Data loader for ALEA Legal Benchmark dataset.
    
    This dataset uses the <|sentence|> marker annotation format.
    Format: text with <|sentence|> markers between sentences
    
    Example:
    {
        "text": "This is the first sentence.<|sentence|> This is the second sentence."
    }
    """
    
    def __init__(self, name: str = "alea"):
        """Initialize the ALEA data loader.
        
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
        """
        super().load(dataset_path, **kwargs)
        
        limit = kwargs.get("limit")
        skip_invalid = kwargs.get("skip_invalid", True)
        
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
                if "text" not in example:
                    if skip_invalid:
                        invalid_examples += 1
                        continue
                    else:
                        raise ValueError(f"Example at line {line_number} is missing 'text' field")
                
                # Add line number for reference
                example["_line_number"] = line_number
                
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
            "format": "ALEA <|sentence|> markers",
            "valid_examples": valid_examples,
            "invalid_examples": invalid_examples,
            "total_lines": line_number
        }
    
    def get_text(self, example: Dict[str, Any]) -> str:
        """Get the raw text from an example.
        
        Args:
            example: Example dictionary
            
        Returns:
            Raw text string with sentence and paragraph markers removed
        """
        if "text" not in example:
            raise ValueError("Example is missing 'text' field")
        
        # Remove sentence and paragraph markers
        return example["text"].replace("<|sentence|>", "").replace("<|paragraph|>", "")
    
    def get_sentences(self, example: Dict[str, Any]) -> List[str]:
        """Get the sentences from an example.
        
        Args:
            example: Example dictionary
            
        Returns:
            List of sentence strings
        """
        if "text" not in example:
            raise ValueError("Example is missing 'text' field")
        
        # Remove paragraph markers first, then split by sentence markers
        text = example["text"].replace("<|paragraph|>", "")
        sentences = text.split("<|sentence|>")
        
        # Normalize sentences
        return self.normalize_sentences(sentences)
    
    def get_spans(self, example: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Get the character spans for each sentence from an example.
        
        Args:
            example: Example dictionary
            
        Returns:
            List of (start, end) tuples representing sentence spans
        """
        if "text" not in example:
            raise ValueError("Example is missing 'text' field")
        
        # First, remove paragraph markers
        text_without_paragraph = example["text"].replace("<|paragraph|>", "")
        
        # Find all sentence marker positions in the text without paragraph markers
        marker = "<|sentence|>"
        marker_positions = []
        pos = text_without_paragraph.find(marker)
        
        while pos != -1:
            marker_positions.append(pos)
            pos = text_without_paragraph.find(marker, pos + 1)
        
        # Create spans from marker positions
        spans = []
        raw_text = self.get_text(example)  # This removes both sentence and paragraph markers
        
        # Handle text before first marker
        start = 0
        
        # Convert marker positions to spans in the raw text
        for marker_pos in marker_positions:
            # Calculate the corresponding position in raw text
            marker_offset = len(marker) * len(spans)  # Account for markers already removed
            raw_end = marker_pos - marker_offset
            
            # Add span for the current sentence
            spans.append((start, raw_end))
            
            # Update start position for the next sentence
            start = raw_end
        
        # Add the final sentence
        if start < len(raw_text):
            spans.append((start, len(raw_text)))
        
        return spans
    
    def normalize_sentences(self, sentences: List[str]) -> List[str]:
        """Normalize sentences to handle punctuation and whitespace.
        
        Args:
            sentences: List of sentence strings
            
        Returns:
            List of normalized sentence strings
        """
        # ALEA specific normalization: trim whitespace and remove paragraph markers
        normalized = []
        for sentence in sentences:
            # Remove paragraph markers
            sentence = sentence.replace("<|paragraph|>", "")
            
            # Skip empty sentences
            if not sentence.strip():
                continue
            
            normalized.append(sentence.strip())
        
        return normalized