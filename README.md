# Precise Legal Sentence Boundary Detection for Retrieval at Scale: NUPunkt and CharBoundary

This repository contains the research, code, and results for our paper on high-precision, high-throughput sentence boundary detection (SBD) libraries optimized for legal text.

## About the Project

Accurate sentence boundary detection is critical for legal document processing, retrieval, and analysis. However, legal text presents unique challenges due to specialized citations, abbreviations, and complex sentence structures that confound general-purpose sentence boundary detectors.

We present two new open-source SBD libraries:

### NUPunkt

A pure Python implementation that extends the unsupervised Punkt algorithm with legal domain optimizations, trained on the KL3M legal corpus.

- **Precision**: 91.1%
- **Throughput**: 10 million characters per second
- **Memory**: 432 MB
- **No external dependencies**
- **29-32% precision improvement** over standard tools like NLTK Punkt and spaCy
- **Links**: [PyPI Package](https://pypi.org/project/nupunkt/) | [GitHub Repository](https://github.com/alea-institute/nupunkt)

### CharBoundary

A family of character-level machine learning models in three sizes (small, medium, large) that offer balanced precision-recall tradeoffs.

- **Highest F1 score**: 0.782 (large model)
- **Throughput**: 518K-748K characters per second depending on model size
- **Requires only scikit-learn** and optional ONNX runtime integration
- **Links**: [PyPI Package](https://pypi.org/project/charboundary/) | [GitHub Repository](https://github.com/alea-institute/charboundary)

## Why This Matters

For legal RAG (Retrieval-Augmented Generation) systems, high precision in sentence boundary detection is essential to prevent fragmentation of related legal concepts, which leads to reasoning failures. Our research shows that the relationship between precision and fragmentation follows an inverse exponential curve, where even small improvements in precision yield significant reductions in downstream errors.

## Experimental Results

Our evaluation on five diverse legal datasets comprising over 25,000 documents and 197,000 annotated sentence boundaries demonstrates that both libraries significantly outperform general-purpose alternatives:

- NUPunkt excels in precision-critical applications where minimizing false positives is paramount
- CharBoundary models provide the best overall F1 scores with excellent balance between precision and recall
- Both libraries enable processing of multi-million document collections in minutes rather than hours on standard CPU hardware

## Datasets

The research utilizes several legal datasets:
- ALEA Legal Benchmark
- MultiLegalSBD (SCOTUS, Cyber Crime, BVA, IP)

## Repository Structure

This repository is organized to help you explore our research, replicate our results, and use our libraries.

| Directory | Description |
|-----------|-------------|
| [`/data`](/data) | Contains the annotated datasets used for evaluation:<br>- `MultiLegalSBD/` - Multiple legal domain datasets with span annotations (SCOTUS, Cyber Crime, BVA, Intellectual Property)<br>- `alea-legal-benchmark/` - ALEA legal benchmark with sentence boundary annotations |
| [`/paper`](/paper) | Complete LaTeX source for the research paper:<br>- `main.tex` - Main paper document<br>- `sections/` - Individual paper sections (introduction, methods, results, etc.)<br>- `figures/` - Publication-quality figures and diagrams<br>- `tables/` - LaTeX tables for benchmark results<br>- `references/` - Bibliography and citations |
| [`/results`](/results) | Comprehensive evaluation results and visualizations:<br>- `paper_results_20250402_203406/` - Latest evaluation results<br>- `evaluation_report.html` - Interactive visualization of results<br>- `charts/` - Performance comparison charts (precision, recall, F1, throughput)<br>- `publication_charts/` - High-quality charts used in the paper<br>- `latex/` - Auto-generated LaTeX tables for the paper |
| [`/src`](/src) | Source code for libraries and evaluation framework:<br>- `lsp/` - Main package (Legal Sentence Paper)<br>- `lsp/tokenizers/` - Implementation of NUPunkt, CharBoundary and baseline tokenizers<br>- `lsp/core/` - Core functionality for data loading and processing<br>- `lsp/examples/` - Example scripts and tools to reproduce experiments<br>- `lsp/evaluation.py` - Evaluation metrics and benchmark logic |

## Getting Started

### Installation

#### Project Installation (for reproducing paper results)

```bash
# Set up Python virtual environment
uv venv --seed && uv pip install pip && source .venv/bin/activate

# Install the project
pip install -e .
```

#### Using the Libraries in Your Projects

Install NUPunkt:
```bash
pip install nupunkt
```

Install CharBoundary:
```bash
pip install charboundary
```
### Usage Examples

Test the tokenizers on legal examples:
```bash
python -m lsp.examples.test_legal_examples
```

Run a complete evaluation:
```bash
python -m lsp.examples.run_evaluation.py --charts --html
```

Process your own text:
```bash
python -m lsp --text "Employee's Annual Bonus shall be calculated pursuant to Sec. 4.3(c), subject to the limitations of I.R.C. § 409A(a)(2)(B)(i) and the withholding requirements of Sec. 7.3." --tokenizers nupunkt charboundary-large
```

### Key CLI Commands

```bash
# List all available datasets
python -m lsp.examples.list_datasets

# Examine a specific dataset
python -m lsp.examples.examine_dataset DATASET [--example ID] [--random N]

# Run paper workflow (reproduces all results)
python -m lsp.examples.paper_workflow.py --output results/
```

For more commands and options, see the [CLAUDE.md](CLAUDE.md) file.

## Demo

Try our interactive demo at [https://sentences.aleainstitute.ai/](https://sentences.aleainstitute.ai/)

## License

Both libraries are available under the MIT license.

## Authors

- Michael J. Bommarito II (ALEA Institute, Stanford CodeX)
- Daniel Martin Katz (Illinois Tech - Chicago Kent Law, Bucerius Law School, ALEA Institute, Stanford CodeX)
- Jillian Bommarito (ALEA Institute)

## Citation

```
@article{bommarito2025nupunkt,
  title={Precise Legal Sentence Boundary Detection for Retrieval at Scale: NUPunkt and CharBoundary},
  author={Bommarito II, Michael J. and Katz, Daniel Martin and Bommarito, Jillian},
  journal={},
  year={2025}
}
```

## Acknowledgments

We drafted and revised this paper with the assistance of large language models. All errors or omissions are our own.