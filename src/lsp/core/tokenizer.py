"""Abstract base class for sentence tokenizers."""

import abc
import time
from typing import Dict, List, Any, Optional, Tuple
import statistics


class SentenceTokenizer(abc.ABC):
    """Abstract base class for sentence tokenizers.
    
    This interface defines a standard way to interact with different sentence
    boundary detection algorithms. Implementations should handle the specifics
    of each library while conforming to this common interface.
    """

    # Standard two-sentence example for warmup and quick testing
    WARMUP_TEXT = "This is a simple first sentence. And here is the second sentence."

    def __init__(self, name: str):
        """Initialize the tokenizer.
        
        Args:
            name: Name identifier for the tokenizer
        """
        self.name = name
        self._is_initialized = False
        self._initialization_time = 0.0
        self._initialization_options = {}
        
    @abc.abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the underlying tokenizer with any required setup.
        
        This method should handle any one-time setup required before tokenization,
        such as loading models, resources, or configuring options.
        
        Args:
            **kwargs: Library-specific initialization options
        """
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if the tokenizer is initialized.
        
        Returns:
            True if the tokenizer is initialized, False otherwise
        """
        return self._is_initialized
    
    @property
    def initialization_time(self) -> float:
        """Get the time taken to initialize the tokenizer.
        
        Returns:
            Time in seconds taken to initialize the tokenizer
        """
        return self._initialization_time
    
    @property
    def initialization_options(self) -> Dict[str, Any]:
        """Get the options used to initialize the tokenizer.
        
        Returns:
            Dictionary of initialization options
        """
        return self._initialization_options.copy()
    
    def warmup(self) -> List[str]:
        """Perform a warmup tokenization on a simple two-sentence example.
        
        This method is useful to ensure that all dependencies are loaded
        and any one-time initializations are complete before benchmarking.
        It can also be used for quick testing of tokenizer functionality.
        
        Returns:
            List of tokenized sentences from the warmup text
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before warmup.")
            
        # Run tokenization on the standard example
        sentences = self.tokenize(self.WARMUP_TEXT)
        
        # Verify we got at least one sentence
        if not sentences:
            raise RuntimeError(f"Warmup failed: Tokenizer {self.name} returned no sentences.")
            
        # Return the sentences for potential testing
        return sentences
    
    @abc.abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into sentences.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of sentence strings
        """
        pass
    
    def get_spans(self, text: str) -> List[Tuple[int, int]]:
        """Get the character spans for each sentence.
        
        This is a derived method that uses tokenize() to determine sentence boundaries
        and then calculates the spans. Implementations can override this for more
        efficient or accurate span detection.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of (start, end) tuples representing sentence spans
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before use.")
        
        # Use tokenize to get sentences
        sentences = self.tokenize(text)
        
        # Calculate spans
        spans = []
        start = 0
        
        for sentence in sentences:
            # Find the sentence in the text starting from the current position
            sentence_start = text.find(sentence, start)
            if sentence_start == -1:
                # This should not happen with correct tokenization
                continue
                
            sentence_end = sentence_start + len(sentence)
            spans.append((sentence_start, sentence_end))
            
            # Update start position for next search
            start = sentence_end
        
        return spans
    
    def benchmark(self, text: str, iterations: int = 3, warmup: bool = True) -> Dict[str, Any]:
        """Benchmark the tokenizer on a text.
        
        Args:
            text: Input text to tokenize
            iterations: Number of iterations to run
            warmup: Whether to perform a warmup run before benchmarking
            
        Returns:
            Dictionary of benchmark results
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before benchmarking.")
        
        # Perform warmup if requested
        if warmup:
            try:
                self.warmup()
            except Exception as e:
                print(f"Warmup for {self.name} failed: {e}")
        
        # Perform timed iterations
        timings = []
        sentence_counts = []
        characters = len(text)
        
        for _ in range(iterations):
            start_time = time.time()
            sentences = self.tokenize(text)
            end_time = time.time()
            
            # Record results
            elapsed = end_time - start_time
            timings.append(elapsed)
            sentence_counts.append(len(sentences))
        
        # Calculate statistics
        avg_time = statistics.mean(timings)
        avg_sentences = statistics.mean(sentence_counts)
        stdev_time = statistics.stdev(timings) if len(timings) > 1 else 0
        throughput = characters / avg_time if avg_time > 0 else 0
        time_per_char = avg_time / characters if characters > 0 else 0
        time_per_sentence = avg_time / avg_sentences if avg_sentences > 0 else 0
        
        return {
            "tokenizer": self.name,
            "characters": characters,
            "iterations": iterations,
            "avg_time_seconds": avg_time,
            "stdev_time_seconds": stdev_time,
            "throughput_chars_per_sec": throughput,
            "time_per_char_seconds": time_per_char,
            "time_per_sentence_seconds": time_per_sentence,
            "sentences": avg_sentences,
            "all_timings": timings,
        }
        
    def benchmark_bulk(self, texts: List[str], iterations: int = 3, warmup_iterations: int = 1) -> Dict[str, Any]:
        """Benchmark the tokenizer on a bulk set of texts (batch processing).
        
        This method provides a more accurate throughput measurement by processing
        all texts in a single batch, similar to how the original libraries
        benchmark themselves.
        
        Args:
            texts: List of texts to tokenize
            iterations: Number of iterations to run
            warmup_iterations: Number of warmup iterations to run
            
        Returns:
            Dictionary of benchmark results
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before benchmarking.")
        
        # Calculate total size of all texts
        total_chars = sum(len(text) for text in texts)
        
        # Perform warmup iterations to engage JIT optimizations
        if warmup_iterations > 0:
            print(f"Performing {warmup_iterations} warmup iteration(s) for {self.name}...")
            for _ in range(warmup_iterations):
                total_sentences = 0
                for text in texts:
                    sentences = self.tokenize(text)
                    total_sentences += len(sentences)
                print(f"  Warmup processed {total_sentences} sentences from {len(texts)} texts")
        
        # Perform timed iterations
        timings = []
        total_sentences_per_iter = []
        
        for i in range(iterations):
            start_time = time.time()
            
            # Process all texts in a batch
            total_sentences = 0
            for text in texts:
                sentences = self.tokenize(text)
                total_sentences += len(sentences)
                
            end_time = time.time()
            
            # Record results
            elapsed = end_time - start_time
            timings.append(elapsed)
            total_sentences_per_iter.append(total_sentences)
            
            print(f"  Iteration {i+1}/{iterations}: "
                  f"{elapsed:.4f}s, "
                  f"{total_sentences} sentences, "
                  f"{total_chars/elapsed:,.0f} chars/sec")
        
        # Calculate statistics
        avg_time = statistics.mean(timings)
        avg_sentences = statistics.mean(total_sentences_per_iter)
        stdev_time = statistics.stdev(timings) if len(timings) > 1 else 0
        throughput = total_chars / avg_time if avg_time > 0 else 0
        time_per_char = avg_time / total_chars if total_chars > 0 else 0
        time_per_sentence = avg_time / avg_sentences if avg_sentences > 0 else 0
        
        # Calculate fastest iteration metrics (most optimized JIT performance)
        min_time_idx = timings.index(min(timings))
        min_time = timings[min_time_idx]
        min_sentences = total_sentences_per_iter[min_time_idx]
        min_throughput = total_chars / min_time if min_time > 0 else 0
        
        return {
            "tokenizer": self.name,
            "text_count": len(texts),
            "total_characters": total_chars,
            "iterations": iterations,
            "avg_time_seconds": avg_time,
            "min_time_seconds": min_time,
            "stdev_time_seconds": stdev_time,
            "throughput_chars_per_sec": throughput,
            "max_throughput_chars_per_sec": min_throughput,
            "time_per_char_seconds": time_per_char,
            "time_per_sentence_seconds": time_per_sentence,
            "total_sentences": avg_sentences,
            "all_timings": timings,
        }
    
    def __str__(self) -> str:
        """Get a string representation of the tokenizer.
        
        Returns:
            String representation of the tokenizer
        """
        status = "initialized" if self.is_initialized else "not initialized"
        return f"{self.name} ({status})"