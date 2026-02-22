SHELL := /bin/bash
.DEFAULT_GOAL:=help
.ONESHELL:
.EXPORT_ALL_VARIABLES:
MAKEFLAGS += --no-print-directory
SHELLFLAGS := -e -o pipefail -c

# Variables
PROJECT_NAME := mogemma

# Formatting
BLUE := $(shell printf "\033[1;34m")
GREEN := $(shell printf "\033[1;32m")
RED := $(shell printf "\033[1;31m")
YELLOW := $(shell printf "\033[1;33m")
NC := $(shell printf "\033[0m")
INFO := $(shell printf "$(BLUE)ℹ$(NC)")
OK := $(shell printf "$(GREEN)✓$(NC)")
ERROR := $(shell printf "$(RED)✖$(NC)")
WARN := $(shell printf "$(YELLOW)⚠$(NC)")

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST) 

.PHONY: install
install: clean ## Install everything (Python, Mojo, Beads)
	@echo "${INFO} Installing..."
	@if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
	@uv python pin 3.10
	@uv venv
	@$(MAKE) py-install
	@$(MAKE) beads-install
	@echo "${OK} Ready!"

.PHONY: beads-install
beads-install: ## Install beads
	@if ! command -v bd >/dev/null 2>&1; then curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash; fi
	@if [ ! -f .beads/config.yaml ]; then bd init --stealth || true; fi

.PHONY: py-install
py-install: ## Install Python deps
	@uv sync --all-extras --dev

.PHONY: build
build: ## Build Mojo shared library
	@echo "${INFO} Building Mojo core..."
	@mkdir -p src/py/mogemma
	@uv run mojo build --emit shared-lib src/mo/core.mojo -o src/py/mogemma/_core.so

.PHONY: smoke-test
smoke-test: ## Run the Mojo bridge smoke test
	@export PYTHONPATH=$PYTHONPATH:$(pwd)/src/py
	@uv run python tools/smoke_test.py

##@ Testing & Quality

.PHONY: test
test: ## Run all tests
	@export PYTHONPATH=$PYTHONPATH:$(pwd)/src/py
	@uv run pytest src/py/tests

.PHONY: lint
lint: ## Lint and format code (Python, Mojo)
	@echo "${INFO} Linting Python (ruff)..."
	@uv run ruff check src/py
	@uv run ruff format --check src/py
	@echo "${INFO} Type checking Python (mypy)..."
	@export PYTHONPATH=$PYTHONPATH:$(pwd)/src/py
	@uv run mypy src/py/mogemma
	@echo "${INFO} Type checking Python (pyright)..."
	@export PYTHONPATH=$PYTHONPATH:$(pwd)/src/py
	@uv run pyright src/py/mogemma
	@echo "${INFO} Formatting Mojo..."
	# @mojo format src/mo
	@echo "${OK} Lint complete"

.PHONY: check-release
check-release: ## Run release preflight checks (lint + tests)
	@echo "${INFO} Running release preflight checks..."
	@echo "${INFO} Linting Python (ruff check)..."
	@uv run ruff check src/py
	@echo "${INFO} Checking Python formatting..."
	@uv run ruff format --check src/py
	@echo "${INFO} Type checking Python (mypy)..."
	@export PYTHONPATH=$PYTHONPATH:$(pwd)/src/py
	@uv run mypy src/py/mogemma
	@echo "${INFO} Type checking Python (pyright)..."
	@export PYTHONPATH=$PYTHONPATH:$(pwd)/src/py
	@uv run pyright src/py/mogemma
	@$(MAKE) test
	@$(MAKE) benchmark
	@echo "${OK} Release preflight checks passed"

.PHONY: type-check
type-check: ## Run all type checkers
	@uv run mypy src/py/$(PROJECT_NAME)
	@uv run pyright src/py/$(PROJECT_NAME)

.PHONY: coverage
coverage: ## Run tests with coverage reports
	@uv run pytest src/py/tests --cov=src/py/$(PROJECT_NAME) --cov-report=html --cov-report=xml
	@echo "${OK} Coverage report: htmlcov/index.html"

.PHONY: benchmark
benchmark: ## Run deterministic release benchmark harness
	@echo "${INFO} Running benchmark (generation)..."
	@uv run python tools/benchmark.py --mode generation --rounds 20 --max-new-tokens 64
	@echo "${INFO} Running benchmark (embedding)..."
	@uv run python tools/benchmark.py --mode embedding --rounds 20
	@echo "${OK} Benchmark complete"

.PHONY: check-all
check-all: lint test coverage ## Run all checks (lint, test, coverage)
	@echo "${OK} All checks passed"

##@ Release

.PHONY: release
release: ## Bump version and create release tag: make release bump=patch
	@echo "${INFO} Preparing for release..."
	@uv run bump-my-version bump $(bump)
	@echo "${OK} Release complete"

.PHONY: pre-release
pre-release: ## Start a pre-release: make pre-release version=0.2.0-alpha.1
	@if [ -z "$(version)" ]; then \
		echo "${ERROR} Usage: make pre-release version=X.Y.Z-alpha.N"; \
		echo "Pre-release workflow:"; \
		echo "  1. Start alpha:     make pre-release version=0.2.0-alpha.1"; \
		echo "  2. Next alpha:      make pre-release version=0.2.0-alpha.2"; \
		echo "  3. Move to beta:    make pre-release version=0.2.0-beta.1"; \
		echo "  4. Move to rc:      make pre-release version=0.2.0-rc.1"; \
		echo "  5. Final release:   make release bump=patch (from rc) OR bump=minor (from stable)"; \
		exit 1; \
	fi
	@echo "${INFO} Preparing pre-release $(version)..."
	@uv run bump-my-version bump --new-version $(version) pre
	@echo "${OK} Pre-release $(version) complete"

##@ Utilities

.PHONY: clean
clean: ## Clean all build artifacts
	@echo "${INFO} Cleaning build artifacts..."
	@rm -rf .venv dist build .pytest_cache .ruff_cache .mypy_cache htmlcov
	@find src/py -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find src/py -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "${OK} Clean complete"

.PHONY: upgrade
upgrade: ## Upgrade all dependencies
	@echo "${INFO} Upgrading dependencies..."
	@uv lock --upgrade
	@echo "${OK} Dependencies upgraded"

.PHONY: destroy
destroy: ## Destroy venv and clean all caches
	@echo "${WARN} Destroying virtual environment..."
	@rm -rf .venv .ruff_cache .mypy_cache .pytest_cache
	@echo "${OK} Environment destroyed. Run 'make install' to recreate."
