# How to set up Python environment
$ uv venv --seed && uv pip install pip && source .venv/bin/activate

# Project Goal
Write a paper for ACL or EMNLP that presents two new open source sentence boundary detection methods (nupunkt and charboundary) and benchmarks them on legal domain text, demonstrating their superior precision and throughput.

## Paper Writing Phase
We are now in the paper writing phase. Your task is to author the LaTeX paper using the resources and results generated in the research phase.


2. **Results**: Use data in `results/paper_results_20250402_203406/`:
   - Complete evaluation metrics in `evaluation_results.json`
   - Interactive visualizations in `evaluation_report.html`
   - Publication-ready charts in `publication_charts/`
   - LaTeX tables already prepared in `latex/`

3. **Library Implementations**:
   - `libraries/nupunkt/`: Pure Python implementation with optimized rules for legal text
   - `libraries/charboundary/`: ML-based implementation using ONNX or scikit-learn

4. **Experiment Code**: 
   - `src/lsp/`: Contains all benchmarking and evaluation code
   - `src/lsp/examples/paper_workflow.py`: Main script that generated our results

## Key Findings to Highlight

## Datasets
Always use these datasets:

### ALEA Legal Benchmark

Format: <|sentence|> annotation

Files:
  - data/alea-legal-benchmark/train.jsonl

Example:

```json lines
{"text": "access to the resources and experience of Bank and its affiliates.<|sentence|>  In addition, there are public benefits to be derived from permitting capital markets to operate so that bank holding companies can make potentially profitable investments in nonbanking companies and from permitting banking organizations to allocate their resources in the manner they consider to be most efficient when such investments are consistent, as in this case, with the relevant considerations under the BHC Act.<|sentence|>  The Board has determined, therefore, that the performance of the proposed activities by"}
```


### MultiLegalSBD 

Format: span annotation

Files:
  - data/MultiLegalSBD/CD_scotus.jsonl
  - data/MultiLegalSBD/CD_cyber_crime.jsonl
  - data/MultiLegalSBD/CD_bva.jsonl
  - data/MultiLegalSBD/CD_intellectual_property.jsonl

Example:

```json lines
{"text":"Apple Computer, Inc. v. Franklin Computer Corporation\nU.S. Court of Appeals Third Circuit\nAugust 30, 1983\n714 F.2d 1240, 219 USPQ 113\n[Editor's note: This case is discussed in Legal Protection of Digital Information in:\nChapter 2, Section II.B.2. (Apple v. Franklin).]\n\nSloviter, Circuit Judge.\n\nI. Introduction\n\nApple Computer, Inc. appeals from the district court’s denial of a motion to preliminarily enjoin Franklin Computer Corp. from infringing the copyrights Apple holds on fourteen computer programs.\nThe decision to grant or refuse to grant a preliminary injunction is within the discretion of the district court. See A.O. Smith Corp. v. FTC, 530 F.2d 515, 525 (3d Cir. 1976). Although the scope of our review of the action of the district court in ruling on a motion for preliminary injunction is narrow, reversal is warranted if the trial court has abused its discretion or committed error in applying the law. Kennecott Corp. v. Smith, 637 F.2d 181, 187 (3d Cir. 1980). As the Second Circuit has stated recently, “Despite oft repeated statements that the issuance of a preliminary injunction rests in the discretion of the trial judge whose decisions will be reversed only for ‘abuse’, a court of appeals must reverse if the district court has proceeded on the basis of an erroneous view of the applicable law.” Donovan v. Bierwirth, 680 F.2d 263, 269 (2d Cir.), cert. denied, 103 S.Ct. 488 (1982).\nIn this case the district court denied the preliminary injunction, inter alia, because it had “some doubt as to the copyrightability of the programs.” Apple Computer, Inc. v. Franklin Computer Corp., 545 F. Supp. 812, 215 USPQ 935 (E.D. Pa. 1982). This legal ruling is fundamental to all future proceedings in this action and, as the parties and amici curiae seem to agree, has considerable significance to the computer services industry.1 Because we conclude that the district court proceeded under an erroneous view of the applicable law, we reverse the denial of the preliminary injunction and ...", "spans":[{"start":0,"end":53,"label":"Sentence","token_start":0,"token_end":9},{"start":54,"end":89,"label":"Sentence","token_start":11,"token_end":19},{"start":90,"end":105,"label":"Sentence","token_start":21,"token_end":24},{"start":106,"end":133,"label":"Sentence","token_start":26,"token_end":35},{"start":134,"end":268,"label":"Sentence","token_start":37,"token_end":73},{"start":270,"end":294,"label":"Sentence","token_start":75,"token_end":79},{"start":296,"end":298,"label":"Sentence","token_start":81,"token_end":82},{"start":299,"end":311,"label":"Sentence","token_start":83,"token_end":83},{"start":313,"end":508,"label":"Sentence","token_start":85,"token_end":118},{"start":509,"end":622,"label":"Sentence","token_start":120,"token_end":139},{"start":623,"end":685,"label":"Sentence","token_start":140,"token_end":167},{"start":686,"end":921,"label":"Sentence","token_start":168,"token_end":210},{"start":922,"end":981,"label":"Sentence","token_start":211,"token_end":233}, ...]}
```
 

# Paper Structure Guidance

## Abstract
- Brief overview of the problem (sentence boundary detection in legal text)
- Introduction to nupunkt and charboundary as solutions
- Summary of key results (precision, recall, throughput)
- Highlight of practical significance

## 1. Introduction
- Importance of sentence boundary detection for legal text processing
- Challenges unique to legal domain (citations, abbreviations, formatting)
- Brief overview of existing approaches and their limitations
- Contributions of this paper:
  - Two new open-source methods optimized for legal text
  - Comprehensive benchmark on diverse legal datasets
  - Analysis of performance/throughput tradeoffs

## 2. Related Work
- Review existing SBD methods (NLTK, spaCy, pySBD)
- Previous work on legal-specific NLP tasks
- Highlight gap in performance for domain-specific applications
- Positioning of nupunkt and charboundary in this landscape

## 3. Methods
- Describe nupunkt algorithm and implementation details
  - Rule-based approach with legal optimization
  - Pure Python implementation
- Describe charboundary models (small, medium, large)
  - Character-level feature extraction
  - ML models (decision trees/random forests)
  - ONNX optimization
- Implementation details focusing on CPU performance

## 4. Experimental Setup
- Dataset descriptions and statistics
  - ALEA Legal Benchmark
  - MultiLegalSBD datasets (SCOTUS, Cyber Crime, BVA, IP)
- Evaluation metrics
  - Precision, recall, F1
  - Time per character, throughput
- Baseline models
  - NLTK punkt
  - spaCy (small and large models)
  - pySBD

## 5. Results and Analysis
- Performance comparison across all datasets
- Error analysis on challenging cases
- Throughput comparison
- Precision/recall tradeoffs
- Ablation studies

## 6. Discussion
- Implications for RAG systems and legal text processing
- Use cases and limitations
- Future work directions

## 7. Conclusion
- Summary of key findings
- Impact on legal NLP applications

## Appendices
- Implementation details
- Additional error analysis
- Data statistics

## CLI Commands

### Main CLI
```bash
python -m lsp [--text TEXT] [--file FILE] [--stdin] [--tokenizers TOKENIZER [...]] [--format {text,json,csv}]
```

### Dataset Tools
```bash
# List all available datasets
python -m lsp.examples.list_datasets

# Examine a specific dataset
python -m lsp.examples.examine_dataset DATASET [--example ID] [--random N] [--analyze] [--sample SIZE] [--limit N] [--export FILE]
```

### Evaluation Tools
```bash
# Run standard evaluation 
python -m lsp.examples.run_evaluation.py [--sample N] [--output DIR] [--charts] [--html] [--html-output PATH] [--max-examples N]

# Complete paper workflow
python -m lsp.examples.paper_workflow.py [--output DIR] [--sample N] [--eval-sample N]

# Benchmark tokenizer throughput
python -m lsp.examples.benchmark_throughput.py [--tokenizers [...]] [--datasets [...]] [--limit N] [--warmup N] [--iterations N] [--output FILE]
```

### Testing Tools
```bash
# Test tokenizers on legal examples
python -m lsp.examples.test_legal_examples

# Evaluate specific tokenizers
python -m lsp.examples.evaluate_tokenizers
```