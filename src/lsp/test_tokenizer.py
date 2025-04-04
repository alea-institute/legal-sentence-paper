"""Test script for tokenizers."""

from lsp.tokenizers.nupunkt import NupunktTokenizer

def main():
    """Run a basic test of the nupunkt tokenizer."""
    # Create a test text with legal content
    test_text = """The Court has recognized that the Fourth Amendment protects citizens against government demands for information. See v. Maryland, 442 U.S. 735, 745-46 (1979).
    In this case, the district court denied the preliminary injunction."""
    
    # Initialize the tokenizer
    tokenizer = NupunktTokenizer()
    tokenizer.initialize()
    
    # Tokenize the text
    sentences = tokenizer.tokenize(test_text)
    
    # Print the results
    print(f"Tokenizer: {tokenizer}")
    print(f"Number of sentences: {len(sentences)}")
    for i, sentence in enumerate(sentences, 1):
        print(f"Sentence {i}: {sentence}")
    
    # Get spans
    spans = tokenizer.get_spans(test_text)
    print("\nSpans:")
    for i, (start, end) in enumerate(spans, 1):
        print(f"Span {i}: ({start}, {end}) = '{test_text[start:end]}'")
    
    # Run benchmark
    benchmark_results = tokenizer.benchmark(test_text, iterations=5)
    print("\nBenchmark results:")
    for key, value in benchmark_results.items():
        if key != "all_timings":  # Skip printing all timings for readability
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()