# Configuration
MAIN_TEX := main.tex
OUTPUT_PDF := $(MAIN_TEX:.tex=.pdf)
BIB_FILE := references/references.bib
LATEX_ENGINE := pdflatex
BIBTEX_ENGINE := bibtex
LATEXMK := latexmk
PYTHON := python

# Directories
FIGURES_DIR := figures
SECTIONS_DIR := sections
TABLES_DIR := tables
RESULTS_DIR := ../results/paper_results_20250402_203406

# Figure generation script (optional)
FIGURE_SCRIPT := ../src/lsp/visualization.py

# Command line options
LATEX_OPTS := -interaction=nonstopmode -halt-on-error
LATEXMK_OPTS := -pdf -bibtex

# Check if we're in verbose mode
ifdef VERBOSE
	QUIET :=
	REDIRECT :=
else
	QUIET := -quiet
	REDIRECT := > /dev/null 2>&1
endif

# Default target - use manual sequence instead of latexmk due to reliability
all: $(OUTPUT_PDF)

# Build using manual sequence which is more reliable
$(OUTPUT_PDF): $(MAIN_TEX) $(wildcard $(SECTIONS_DIR)/*.tex) $(wildcard $(TABLES_DIR)/*.tex) $(wildcard $(FIGURES_DIR)/*.pdf) $(BIB_FILE)
	@echo "Building $(OUTPUT_PDF)..."
	$(LATEX_ENGINE) $(LATEX_OPTS) $(MAIN_TEX)
	$(BIBTEX_ENGINE) $(basename $(MAIN_TEX))
	$(LATEX_ENGINE) $(LATEX_OPTS) $(MAIN_TEX)
	$(LATEX_ENGINE) $(LATEX_OPTS) $(MAIN_TEX)
	@echo "Build complete."

# Alternative build using latexmk (may fail more often)
latexmk:
	@echo "Building with latexmk..."
	$(LATEXMK) $(LATEXMK_OPTS) $(MAIN_TEX)
	@echo "Build complete."

# Copy publication figures from results
figures:
	@echo "Copying publication figures..."
	mkdir -p $(FIGURES_DIR)
	cp $(RESULTS_DIR)/publication_charts/*.pdf $(FIGURES_DIR)/
	@echo "Figures copied."

# Copy LaTeX tables from results directory
tables:
	@echo "Copying LaTeX tables..."
	mkdir -p $(TABLES_DIR)
	cp $(RESULTS_DIR)/latex/*.tex $(TABLES_DIR)/
	# Fix underscores in table files
	# sed -i 's/alea_train/ALEA Train/g; s/multilegal_scotus/SCOTUS/g; s/multilegal_cyber_crime/Cyber Crime/g; s/multilegal_bva/BVA/g; s/multilegal_intellectual_property/Intellectual Property/g' $(TABLES_DIR)/dataset_stats_table.tex
	# sed -i 's/alea_train/ALEA Train/g; s/multilegal_scotus/SCOTUS/g; s/multilegal_cyber_crime/Cyber Crime/g; s/multilegal_bva/BVA/g; s/multilegal_intellectual_property/Intellectual Property/g; s/charboundary_/charboundary-/g; s/nltk_punkt/nltk punkt/g; s/spacy_sm/spaCy (sm)/g; s/spacy_lg/spaCy (lg)/g; s/pysbd/pySBD/g' $(TABLES_DIR)/performance_table.tex
	@echo "Tables copied and fixed."

# View the compiled PDF
view: $(OUTPUT_PDF)
	@echo "Opening $(OUTPUT_PDF)..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(OUTPUT_PDF); \
	elif command -v open > /dev/null; then \
		open $(OUTPUT_PDF); \
	else \
		echo "Could not find a program to open $(OUTPUT_PDF)"; \
	fi

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	@echo "Clean complete."

# Full rebuild
rebuild: clean all

# Prepare assets (figures and tables)
prepare: figures tables

# Complete build process including assets
full: clean prepare 
	@echo "Running full build with complete sequence..."
	$(LATEX_ENGINE) $(LATEX_OPTS) $(MAIN_TEX)
	$(BIBTEX_ENGINE) $(basename $(MAIN_TEX))
	$(LATEX_ENGINE) $(LATEX_OPTS) $(MAIN_TEX)
	$(LATEX_ENGINE) $(LATEX_OPTS) $(MAIN_TEX)
	@echo "Full build complete."

# Check status of required directories and files
check:
	@echo "Checking paper structure..."
	@mkdir -p $(FIGURES_DIR) $(TABLES_DIR)
	@echo "Main tex file: $(shell [ -f $(MAIN_TEX) ] && echo 'Found' || echo 'Missing')"
	@echo "Bibliography file: $(shell [ -f $(BIB_FILE) ] && echo 'Found' || echo 'Missing')"
	@echo "Sections directory: $(shell [ -d $(SECTIONS_DIR) ] && echo 'Found' || echo 'Missing')"
	@echo "Tables directory: $(shell [ -d $(TABLES_DIR) ] && echo 'Found' || echo 'Missing')"
	@echo "Figures directory: $(shell [ -d $(FIGURES_DIR) ] && echo 'Found' || echo 'Missing')"
	@echo "Section files found: $(shell ls -1 $(SECTIONS_DIR)/*.tex 2>/dev/null | wc -l)"
	@echo "Table files found: $(shell ls -1 $(TABLES_DIR)/*.tex 2>/dev/null | wc -l)"
	@echo "Figure files found: $(shell ls -1 $(FIGURES_DIR)/*.pdf 2>/dev/null | wc -l)"

# Help target
help:
	@echo "Legal Sentence Boundary Detection Paper Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all       - Build the PDF using manual sequence (default target)"
	@echo "  latexmk   - Build using latexmk (alternative)"
	@echo "  clean     - Remove temporary files"
	@echo "  rebuild   - Clean and rebuild"
	@echo "  view      - Open the PDF with default viewer"
	@echo "  figures   - Copy publication figures from results"
	@echo "  tables    - Copy LaTeX tables from results"
	@echo "  prepare   - Copy all assets (figures and tables)"
	@echo "  full      - Complete build including asset copying"
	@echo "  check     - Check paper structure and required files"
	@echo ""
	@echo "Options:"
	@echo "  VERBOSE=1 - Show detailed build output"

# Phony targets
.PHONY: all latexmk clean rebuild view figures tables prepare full check help