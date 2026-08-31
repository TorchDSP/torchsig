# ==============================================================================
# TorchSig Project Makefile
# ==============================================================================

# --- Variables ---
PYTHON = python3
PIP = $(PYTHON) -m pip
PYTEST = $(PYTHON) -m pytest

# Directories
SRC_DIR = torchsig
TEST_DIR = tests
# Cap local xdist workers
# override with:  PYTEST_NPROCS=8 make test
PYTEST_NPROCS ?= 2

# --- Phony Targets ---
# .PHONY tells Make that these are "commands" and not "files" on disk.
# This prevents conflicts if you have a folder named 'tests' or 'clean'.
.PHONY: install build verify publish test test-debug test-cov test-notebooks test-notebooks-clean \
		clean-notebooks lint format fix clean docs open-docs help benchmarks benchmarks-clean check-lfs

# Default target: show help
help:
	@echo "TorchSig Project Orchestration"
	@echo "-----------------------------"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install    Install dependencies and the package in editable mode"
	@echo "  build      Builds source distribution (sdist) and wheel for PyPI"
	@echo "  verify     Validates distribution files and lists them (pre-publish check)"
	@echo "  publish    Uploads distribution files to PyPI"
	@echo "  test       Run all tests in the $(TEST_DIR) directory in parallel"
	@echo "  test-debug Run all tests in the $(TEST_DIR) directory sequentially"
	@echo "  test-cov   Run tests and generate a coverage report"
	@echo "  test-notebooks"
	@echo "             Executes all Jupyter notebooks to verify they run without errors."
	@echo "  test-notebooks-clean"
	@echo "             Removes stamp files created by notebook execution"
	@echo "  clean-notebooks"
	@echo "             Removes all output from executed notebooks"
	@echo "  lint       Run static analysis (ruff check)"
	@echo "  format     Auto-format code (ruff format)"
	@echo "  fix        Apply automatic fixes and formatting (ruff check --fix)"
	@echo "  docs       Builds the HTML documentation"
	@echo "  open-docs  Opens the built documentation in the default browser"
	@echo "  benchmarks Runs the benchmarks"
	@echo "  benchmarks-clean"
	@echo "             Removes previous benchmark results"
	@echo "  check-lfs  Checks for any remaining Git LFS references in the repository"
	@echo "  clean      Remove caches and temporary test artifacts"
	@echo "  help       Show this help message"

# --- Targets ---

# Setup
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"


# Build
build: .build-stamp

.build-stamp:
	pip install --upgrade build
	python -m build --sdist --wheel

verify: build
	twine check dist/*
	ls -lh dist/

publish: verify
	twine check dist/*
	twine upload dist/*


# Testing
TEST_MODE ?= fast

# Report configuration
JUNIT_REPORT ?= report.xml
COVERAGE_XML ?= coverage.xml
COVERAGE_TERM ?= term-missing

test:
	@echo "Running tests ($(TEST_MODE) mode) with $(PYTEST_NPROCS) workers..."
	$(PYTEST) \
		-n $(PYTEST_NPROCS) \
		--dist=loadfile \
		--ignore=benchmarks \
		--test-mode=$(TEST_MODE) \
		--junitxml=$(JUNIT_REPORT) \
		$(TEST_DIR)

test-debug:
	@echo "Running tests (debug sequential)..."
	$(PYTEST) \
		-n 0 \
		--junitxml=$(JUNIT_REPORT) \
		$(TEST_DIR)

test-cov:
	@echo "Running tests with coverage (parallel)..."
	$(PYTEST) \
		-n $(PYTEST_NPROCS) \
		--dist=loadfile \
		--ignore=benchmarks \
		--test-mode=$(TEST_MODE) \
		--junitxml=$(JUNIT_REPORT) \
		--cov=$(SRC_DIR) \
		--cov-report=xml:$(COVERAGE_XML) \
		--cov-report=$(COVERAGE_TERM) \
		$(TEST_DIR)

# test-notebooks target
HAS_GPU := $(shell python3 -c "import torch; print(int(torch.cuda.is_available()))")

ifeq ($(HAS_GPU),1)
NOTEBOOKS := $(wildcard examples/*.ipynb examples/*/*.ipynb)
else
NOTEBOOKS := $(filter-out \
	examples/classifier_example.ipynb \
	examples/create_dataset_example.ipynb, \
	$(wildcard examples/*.ipynb examples/*/*.ipynb))
endif
EXECUTED_STAMPS  := $(patsubst %.ipynb,%.executed,$(NOTEBOOKS))

test-notebooks: $(EXECUTED_STAMPS)

%.executed: %.ipynb
	jupyter trust $< || { echo "❌ $< is not a valid notebook"; exit 1; }
	PYTHONPATH=$(PWD) jupyter nbconvert \
		--to notebook \
		--execute $< \
		--output $(notdir $<) \
		--output-dir $(dir $<) \
		--ExecutePreprocessor.startup_timeout=300 \
		--ExecutePreprocessor.timeout=7200 \
		|| { rm -f $@; exit 1; }
	touch $@

test-notebooks-clean:
	rm -f $(EXECUTED_STAMPS) .cleaned


clean-notebooks:
	echo "🧽  Running nbclean on all notebooks…"
	nb-clean clean $(NOTEBOOKS) || { echo "❌ nb-clean failed – see the messages above"; exit 1; }

# The *file* that tells make the cleaning has finished.
# It is touched only after the clean step succeeded.
.cleaned: clean-notebooks
	touch $@


# Quality Assurance
lint:
	@echo "Running Ruff check..."
	$(PYTHON) -m ruff check $(SRC_DIR)

format:
	@echo "Formatting with Ruff..."
	$(PYTHON) -m ruff format $(SRC_DIR)

fix:
	@echo "Automatically fixing lint errors..."
	$(PYTHON) -m ruff check --fix $(SRC_DIR)
	$(PYTHON) -m ruff format $(SRC_DIR)


# Docs
docs:
	cd docs && make html && cd ..

open-docs: docs
	echo "Opening docs in browser..."
	xdg-open docs/build/html/index.html || \
		open docs/build/html/index.html || \
		start docs/build/html/index.html || \
		firefox docs/build/html/index.html

# Benchmarks
BENCHMARK_FILES = \
	benchmarks/benchmark_transforms_functional.py \
	benchmarks/benchmark_dataset_generation.py
BENCHMARK_DIR ?= .benchmarks
BENCHMARK_OUTPUT ?= $(BENCHMARK_DIR)/benchmark_output.txt
BENCHMARK_JSON ?= $(BENCHMARK_DIR)/benchmark.json

benchmarks:  ## Run performance benchmarks
	@echo "Running benchmarks..."
	mkdir -p $(BENCHMARK_DIR)
	$(PYTEST) $(BENCHMARK_FILES) \
		--benchmark-only \
		--benchmark-save=$(BENCHMARK_DIR) \
		--benchmark-json=$(BENCHMARK_JSON) \
		--benchmark-histogram \
		--benchmark-compare \
		-v \
		--no-cov 2>&1 | tee $(BENCHMARK_OUTPUT)


benchmarks-clean:  ## Remove benchmark data
	@echo "Cleaning benchmark data..."
	rm -rf $(BENCHMARK_DIR)


# Maintenance
clean:
	@echo "Cleaning up..."
	# Remove Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	# Remove pytest and coverage artifacts
	rm -rf .pytest_cache .coverage htmlcov
	# Remove temporary directories created by file-handler tests
	# (Adjust the pattern if your tests use a different tmp naming convention)
	rm -fr "/tmp/pytest-of-$(USER)"
	@echo "Cleanup complete."

check-lfs:
	@echo "🔍 Checking branch: $(shell git rev-parse --abbrev-ref HEAD) (post filter-repo)"
	@LFS_FOUND=0; \
	# 1. Check current .gitattributes \
	printf "\n1. Checking current .gitattributes...\n"; \
	if [ -f .gitattributes ]; then \
		if grep -q "filter=lfs" .gitattributes; then \
			printf "\033[0;31m   ❌ Current .gitattributes contains LFS rules\033[0m\n"; \
			LFS_FOUND=1; \
		else \
			printf "\033[0;32m   ✅ Current .gitattributes has no LFS rules\033[0m\n"; \
		fi; \
	else \
		printf "\033[0;32m   ✅ No .gitattributes file exists\033[0m\n"; \
	fi; \
	# 2. Check .gitattributes history \
	printf "\n2. Checking .gitattributes history...\n"; \
	if git log --all --oneline -- .gitattributes >/dev/null 2>&1; then \
		printf "   Found .gitattributes in history...\n"; \
		git log --all --pretty=format:'%H' -- .gitattributes | while read -r commit; do \
			if git show "$$commit:.gitattributes" 2>/dev/null | grep -q "filter=lfs"; then \
				SHORT_COMMIT=$${commit:0:7}; \
				printf "\033[0;31m   ❌ LFS rule found in .gitattributes (commit: $$SHORT_COMMIT)\033[0m\n"; \
				LFS_FOUND=1; \
			fi; \
		done; \
	else \
		printf "\033[0;32m   ✅ No .gitattributes in history\033[0m\n"; \
	fi; \
	# 3. Check for LFS pointers in current files \
	printf "\n3. Checking current files for LFS pointers...\n"; \
	git ls-tree -r HEAD --name-only 2>/dev/null | while read -r file; do \
		if [ "$$file" = ".gitattributes" ]; then continue; fi; \
		blob=$(git rev-parse "HEAD:$$file" 2>/dev/null); \
		if [ -n "$$blob" ]; then \
			content=$(git cat-file -p "$$blob" 2>/dev/null | head -1); \
			if [[ "$$content" == _* ]]; then \
				printf "\033[0;31m   ❌ LFS pointer found in current file: $$file\033[0m\n"; \
				LFS_FOUND=1; \
			fi; \
		fi; \
	done; \
	if [ "$$LFS_FOUND" -eq 0 ]; then \
		printf "\033[0;32m   ✅ No LFS pointers in current files\033[0m\n"; \
	fi; \
	# 4. Check Git LFS cache \
	printf "\n4. Checking Git LFS cache...\n"; \
	if [ -d .git/lfs/objects ]; then \
		LFS_OBJECTS=$$(find .git/lfs/objects -type f 2>/dev/null | wc -l); \
		if [ "$$LFS_OBJECTS" -gt 0 ]; then \
			printf "\033[0;31m   ❌ Found $$LFS_OBJECTS LFS objects in .git/lfs/objects\033[0m\n"; \
			LFS_FOUND=1; \
		else \
			printf "\033[0;32m   ✅ No LFS objects in cache\033[0m\n"; \
		fi; \
	else \
		printf "\033[0;32m   ✅ No LFS cache directory\033[0m\n"; \
	fi; \
	# 5. Check Git config \
	printf "\n5. Checking Git config...\n"; \
	if git config --get-regexp 'filter.lfs' >/dev/null 2>&1; then \
		printf "\033[0;31m   ❌ LFS filter found in Git config\033[0m\n"; \
		LFS_FOUND=1; \
		git config --get-regexp 'filter.lfs' | while read -r key value; do \
			printf "\033[0;31m      $$key=$$value\033[0m\n"; \
		done; \
	else \
		printf "\033[0;32m   ✅ No LFS in Git config\033[0m\n"; \
	fi; \
	# Final result \
	printf "\n=== Summary for $(shell git rev-parse --abbrev-ref HEAD) (post filter-repo) ===\n"; \
	if [ "$$LFS_FOUND" -eq 0 ]; then \
		printf "\033[0;32m✅ No active Git LFS objects or references found\033[0m\n"; \
		printf "\033[0;32m   (History may have had LFS, but it's been cleaned)\033[0m\n"; \
		exit 0; \
	else \
		printf "\033[0;31m❌ Active Git LFS references still exist\033[0m\n"; \
		printf "\033[1;33m   To fix:\033[0m\n"; \
		printf "\033[1;33m   1. Remove LFS config: git config --remove-section filter.lfs\033[0m\n"; \
		printf "\033[1;33m   2. Prune LFS cache: git lfs prune\033[0m\n"; \
		printf "\033[1;33m   3. If needed, run: git filter-repo --invert-paths --path .gitattributes\033[0m\n"; \
		exit 1; \
	fi



# ==============================================================================
# Notes:
# - Use 'make install' when first setting up the repo.
# - Use 'make test' before every commit.
# - Use 'make clean' if you encounter weird filesystem issues.
# ==============================================================================

