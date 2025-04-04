"""spaCy implementation of SentenceTokenizer."""

import time
import warnings
from typing import List, Dict, Any, Optional, Tuple

from lsp.core.tokenizer import SentenceTokenizer

# Filter out the W108 lemmatizer warning from spaCy
warnings.filterwarnings("ignore", message=".*W108.*lemmatizer did not find POS annotation.*")


class SpacyTokenizer(SentenceTokenizer):
    """SentenceTokenizer implementation for spaCy.
    
    spaCy provides rule-based and statistical sentence boundary detection
    with various pre-trained models.
    """
    
    def __init__(self, name: str = "spacy"):
        """Initialize the spaCy tokenizer.
        
        Args:
            name: Name identifier for the tokenizer
        """
        super().__init__(name)
        self._nlp = None
        
    def initialize(self, **kwargs) -> None:
        """Initialize the spaCy tokenizer.
        
        Args:
            **kwargs: Initialization options:
                - model: spaCy model to use (default: 'en_core_web_sm')
                - disable: Components to disable (default: ['ner', 'parser', 'attribute_ruler'])
        """
        import spacy
        
        start_time = time.time()
        
        model = kwargs.get("model", "en_core_web_sm")
        # Disable unnecessary components for better performance
        disable = kwargs.get("disable", ["ner", "attribute_ruler"])
        
        # Try to load the requested spaCy model
        try:
            # For transformer models, keep the parser for sentence boundaries
            # For other models, use a dedicated sentencizer
            if "trf" in model or "lg" in model:
                self._nlp = spacy.load(model, disable=disable)
            else:
                # For non-transformer models, we'll use the sentencizer
                # Do not disable the parser if it's explicitly requested to be enabled
                if "parser" in disable:
                    disable.remove("parser")
                self._nlp = spacy.load(model, disable=disable)
                # Add sentencizer if parser is not present
                if "parser" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("sentencizer")
        except Exception as e:
            # If model loading fails, create a blank English model with a sentencizer
            print(f"Warning: Could not load spaCy model {model}: {e}")
            print(f"Creating blank model with sentencizer for {model}")
            
            self._nlp = spacy.blank("en")
            # Add a sentencizer component for sentence boundary detection
            if "sentencizer" not in self._nlp.pipe_names:
                self._nlp.add_pipe("sentencizer")
        
        end_time = time.time()
        self._initialization_time = end_time - start_time
        self._initialization_options = kwargs.copy()
        self._is_initialized = True
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into sentences using spaCy.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of sentence strings
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before use.")
        
        doc = self._nlp(text)
        return [sent.text for sent in doc.sents]
        
    def get_spans(self, text: str) -> List[Tuple[int, int]]:
        """Get the character spans for each sentence using spaCy.
        
        This is more efficient than the base class implementation as it
        directly uses spaCy's span information.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of (start, end) tuples representing sentence spans
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before use.")
        
        doc = self._nlp(text)
        return [(sent.start_char, sent.end_char) for sent in doc.sents]