"""NLTK Punkt implementation of SentenceTokenizer."""

import time
from typing import List, Dict, Any, Optional

from lsp.core.tokenizer import SentenceTokenizer


class NLTKTokenizer(SentenceTokenizer):
    """SentenceTokenizer implementation for NLTK's Punkt tokenizer.
    
    The NLTK Punkt tokenizer is an unsupervised multilingual sentence boundary detector
    that uses a pre-trained model to identify sentence boundaries.
    """
    
    def __init__(self, name: str = "nltk_punkt"):
        """Initialize the NLTK Punkt tokenizer.
        
        Args:
            name: Name identifier for the tokenizer
        """
        super().__init__(name)
        self._tokenizer = None
        
    def initialize(self, **kwargs) -> None:
        """Initialize the NLTK Punkt tokenizer.
        
        Args:
            **kwargs: Initialization options:
                - language: Language model to use (default: 'english')
                - train_text: Optional text to train on for a custom model
        """
        import nltk.tokenize
        from nltk.tokenize.punkt import PunktSentenceTokenizer
        import nltk.data
        
        start_time = time.time()
        
        # Ensure the punkt model is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        language = kwargs.get("language", "english")
        train_text = kwargs.get("train_text")
        
        # Import the sent_tokenize function directly
        from nltk.tokenize import sent_tokenize
        
        # Try to use the proper NLTK tokenizer
        try:
            # Create a tokenizer function
            self._tokenizer = lambda text: sent_tokenize(text, language=language)
            # Test it to make sure it works
            test_result = self._tokenizer("This is a test. This is another test.")
            if len(test_result) != 2:
                raise RuntimeError("NLTK tokenizer not working properly")
                
        except Exception as e:
            # If we can't use the NLTK tokenizer, we should fail initialization
            raise RuntimeError(f"Could not initialize NLTK Punkt tokenizer: {e}")
            
        self._language = language
        
        end_time = time.time()
        self._initialization_time = end_time - start_time
        self._initialization_options = kwargs.copy()
        self._is_initialized = True
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into sentences using NLTK Punkt.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of sentence strings
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before use.")
        
        # We always use the tokenizer function directly now
        return self._tokenizer(text)