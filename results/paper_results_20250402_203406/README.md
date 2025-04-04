# Legal Sentence Boundary Detection Results

Results generated on: 2025-04-02 20:44:26

## Contents

- `evaluation_results.json`: Complete evaluation results
- `evaluation_report.html`: Interactive HTML report with color-coded sentence analysis
- `dataset_stats.json`: Statistics for all datasets
- `tokenizers.json`: Information about tokenizers used

### Directories

- `charts/`: Visualizations of tokenizer performance
- `stats/`: Detailed statistics for each dataset
- `latex/`: LaTeX tables for the paper
- `publication_charts/`: Publication-quality charts for ACL/EMNLP paper

## Summary

- Tokenizers evaluated: 8
- Datasets analyzed: 5
- Evaluation sample size: 10000

To regenerate these results, run:
```
python src/lsp/examples/paper_workflow.py --output results --sample 100000 --eval-sample 10000
```
