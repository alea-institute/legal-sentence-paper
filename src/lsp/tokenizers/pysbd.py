"""PySBD implementation of SentenceTokenizer."""

import time
from typing import List, Dict, Any, Optional, Tuple

from lsp.core.tokenizer import SentenceTokenizer


class PySBDTokenizer(SentenceTokenizer):
    """SentenceTokenizer implementation for PySBD.
    
    PySBD is a rule-based sentence boundary detection library,
    ported from Ruby's Pragmatic Segmenter.
    """
    
    def __init__(self, name: str = "pysbd"):
        """Initialize the PySBD tokenizer.
        
        Args:
            name: Name identifier for the tokenizer
        """
        super().__init__(name)
        self._segmenter = None
        
    def initialize(self, **kwargs) -> None:
        """Initialize the PySBD tokenizer.
        
        Args:
            **kwargs: Initialization options:
                - language: Language to use (default: 'en')
                - clean: Whether to apply cleaning rules (default: True)
                - char_span: Whether to include character spans (default: False)
        """
        import pysbd
        
        start_time = time.time()
        
        language = kwargs.get("language", "en")
        clean = kwargs.get("clean", True)
        char_span = kwargs.get("char_span", False)
        
        self._segmenter = pysbd.Segmenter(language=language, clean=clean, char_span=char_span)
        self._char_span = char_span
        
        end_time = time.time()
        self._initialization_time = end_time - start_time
        self._initialization_options = kwargs.copy()
        self._is_initialized = True
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into sentences using PySBD.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of sentence strings
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before use.")
        
        try:
            if self._char_span:
                # If char_span is True, we need to extract just the text
                segments = self._segmenter.segment(text)
                return [segment.sent for segment in segments]
            else:
                # Otherwise, segment directly returns list of strings
                return self._segmenter.segment(text)
        except Exception:
            # If tokenization fails, return empty list (0% correct)
            return []
        
    def get_spans(self, text: str) -> List[Tuple[int, int]]:
        """Get the character spans for each sentence using PySBD.
        
        If initialized with char_span=True, this method will use PySBD's
        built-in span information. Otherwise, it will fall back to the
        base class implementation.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of (start, end) tuples representing sentence spans
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before use.")
        
        try:
            if self._char_span:
                # If we initialized with char_span=True, we can get spans directly
                segments = self._segmenter.segment(text)
                return [(segment.start, segment.end) for segment in segments]
            else:
                # Otherwise, use the base class implementation
                return super().get_spans(text)
        except Exception:
            # If getting spans fails, return empty list
            return []