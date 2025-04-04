"""Generate a sample HTML report from a small evaluation."""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lsp.visualization_html import generate_html_report

def main():
    """Generate a sample HTML report."""
    # Path to use - either from command line or default
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = "results/small_test.json"
    
    if not os.path.exists(results_path):
        print(f"Error: Results file {results_path} does not exist.")
        print("Please run a small evaluation first or specify a valid results file path.")
        return
    
    # Load the results
    try:
        with open(results_path, "r") as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}")
        return
    
    # Generate HTML report
    html_output = results_path.replace(".json", ".html")
    try:
        generate_html_report(results, html_output, max_examples_per_dataset=2)
        print(f"Sample HTML report generated at {html_output}")
    except Exception as e:
        print(f"Error generating HTML report: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()