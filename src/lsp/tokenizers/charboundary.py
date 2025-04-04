"""CharBoundary implementation of SentenceTokenizer."""

import time
import os
from typing import List, Dict, Any, Optional, Tuple

from lsp.core.tokenizer import SentenceTokenizer


class CharBoundaryTokenizer(SentenceTokenizer):
    """SentenceTokenizer implementation for CharBoundary.
    
    CharBoundary is a machine learning-based sentence boundary detector
    optimized for legal and scientific texts.
    """
    
    def __init__(self, name: str = "charboundary"):
        """Initialize the CharBoundary tokenizer.
        
        Args:
            name: Name identifier for the tokenizer
        """
        super().__init__(name)
        self._model = None
        
    def initialize(self, **kwargs) -> None:
        """Initialize the CharBoundary tokenizer.
        
        Args:
            **kwargs: Initialization options:
                - size: Model size to use ('small', 'medium', 'large', default: 'small')
                - model_path: Optional path to a custom model file
        """
        import charboundary
        
        start_time = time.time()
        
        size = kwargs.get("size", "small")
        model_path = kwargs.get("model_path")
        
        if model_path and os.path.exists(model_path):
            # Load a custom model from path if feature is added in future versions
            raise ValueError("Custom model loading not supported in current charboundary version")
        else:
            # Load a pre-trained ONNX model based on size
            if size == "small":
                self._model = charboundary.get_small_onnx_segmenter()
            elif size == "large":
                self._model = charboundary.get_large_onnx_segmenter()
            else:  # medium or default
                self._model = charboundary.get_medium_onnx_segmenter()
        
        end_time = time.time()
        self._initialization_time = end_time - start_time
        self._initialization_options = kwargs.copy()
        self._is_initialized = True
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into sentences using CharBoundary.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of sentence strings
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before use.")
        
        # Using the preferred segment_to_sentences method from README
        try:
            # First try the preferred method from documentation
            sentences = self._model.segment_to_sentences(text)
            return sentences
        except AttributeError:
            # Fallback to segment_text if segment_to_sentences is not available
            # This handles any version differences
            segmented = self._model.segment_text(text)
            
            # Check if the result is a string with <|sentence|> markers
            if isinstance(segmented, str):
                # Split by the sentence marker and filter out any empty strings
                return [s for s in segmented.split("<|sentence|>") if s]
            elif isinstance(segmented, list):
                # Handle possible list return format
                return segmented
            else:
                # Fallback to original text
                return [text]
        
    def get_spans(self, text: str) -> List[Tuple[int, int]]:
        """Get the character spans for each sentence using CharBoundary.
        
        For CharBoundary, we need to determine spans from the tokenization results
        since the API doesn't directly provide spans.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of (start, end) tuples representing sentence spans
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before use.")
        
        # Use the base implementation to calculate spans from sentences
        return super().get_spans(text)