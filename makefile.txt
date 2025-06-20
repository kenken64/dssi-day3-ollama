# Text-to-SQL Ollama Model Makefile
# Provides convenient commands for development and deployment

.PHONY: help install setup train deploy test clean docker lint format check-requirements

# Default target
help: ## Show this help message
	@echo "Text-to-SQL Ollama Model - Available Commands:"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation and Setup
install: ## Install all dependencies
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully"

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	@echo "✅ Development dependencies installed"

setup: ## Setup project structure and check requirements
	python -c "from utils import create_directory_structure, check_system_requirements; create_directory_structure(); print('System Requirements:'); [print(f'  {k}: {"✓" if v else "✗"}') for k, v in check_system_requirements().items()]"
	@echo "✅ Project setup complete"

check-requirements: ## Check system requirements
	@python -c "from utils import check_system_requirements; reqs = check_system_requirements(); print('System Requirements Check:'); [print(f'  {k}: {\"✓\" if v else \"✗\"}') for k, v in reqs.items()]; exit(0 if all(reqs.values()) else 1)"

# Training and Model Management
train: ## Train the model with default settings
	python text_to_sql_train.py --mode train --max-samples 5000
	@echo "✅ Model training completed"

train-quick: ## Quick training with limited samples for testing
	python text_to_sql_train.py --mode train --max-samples 1000
	@echo "✅ Quick training completed"

train-full: ## Full training with complete dataset
	python text_to_sql_train.py --mode full
	@echo "✅ Full training and deployment completed"

train-sqlcoder: ## Train using SQLCoder base model
	python text_to_sql_train.py --mode train --base-model "defog/sqlcoder-7b"
	@echo "✅ SQLCoder model training completed"

train-codellama: ## Train using CodeLlama base model
	python text_to_sql_train.py --mode train --base-model "codellama/CodeLlama-7b-Instruct-hf"
	@echo "✅ CodeLlama model training completed"

# Deployment
deploy: ## Deploy model to Ollama
	python deploy.py --config config.yaml
	@echo "✅ Model deployed to Ollama"

deploy-force: ## Force redeploy model to Ollama
	python deploy.py --config config.yaml --force
	@echo "✅ Model force redeployed to Ollama"

deploy-test: ## Deploy and run tests
	python deploy.py --config config.yaml --test
	@echo "✅ Model deployed and tested"

deploy-demo: ## Deploy and run interactive demo
	python deploy.py --config config.yaml --demo
	@echo "✅ Interactive demo started"

# Testing
test: ## Run comprehensive test suite
	python test.py --config config.yaml
	@echo "✅ Tests completed"

test-quick: ## Run quick test subset
	python test.py --config config.yaml --quick
	@echo "✅ Quick tests completed"

test-performance: ## Run performance tests only
	python test.py --config config.yaml --performance-only
	@echo "✅ Performance tests completed"

# Code Quality
lint: ## Run linting checks
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "✅ Linting completed"

format: ## Format code with black
	black .
	@echo "✅ Code formatting completed"

format-check: ## Check code formatting
	black --check .
	@echo "✅ Code formatting check completed"

# Docker Operations
docker-build: ## Build Docker image
	docker build -t text-to-sql-trainer .
	@echo "✅ Docker image built"

docker-run: ## Run training in Docker container
	docker run --gpus all -v $(PWD):/workspace text-to-sql-trainer python text_to_sql_train.py --mode train
	@echo "✅ Docker training completed"

docker-up: ## Start Docker Compose services
	docker-compose up -d
	@echo "✅ Docker services started"

docker-down: ## Stop Docker Compose services
	docker-compose down
	@echo "✅ Docker services stopped"

docker-logs: ## View Docker logs
	docker-compose logs -f

# Utility Commands
clean: ## Clean up generated files and caches
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf models/*/
	rm -rf processed_data/
	rm -rf cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "✅ Cleanup completed"

clean-models: ## Clean up trained models
	rm -rf models/
	rm -rf processed_data/
	@echo "✅ Model files cleaned up"

clean-logs: ## Clean up log files
	rm -rf logs/
	rm -f *.log
	@echo "✅ Log files cleaned up"

# Data Management
download-dataset: ## Download and cache the dataset
	python -c "from datasets import load_dataset; load_dataset('gretelai/synthetic_text_to_sql', cache_dir='./cache')"
	@echo "✅ Dataset downloaded and cached"

analyze-dataset: ## Analyze the dataset
	python -c "from utils import DatasetAnalyzer; from datasets import load_dataset; analyzer = DatasetAnalyzer(); dataset = load_dataset('gretelai/synthetic_text_to_sql')['train']; import json; print(json.dumps(analyzer.generate_analysis_report(dataset), indent=2))"

# Development Helpers
dev-setup: install-dev setup ## Complete development setup
	@echo "✅ Development environment ready"

quick-start: check-requirements train-quick deploy-test ## Quick start for testing
	@echo "✅ Quick start completed - model ready for testing"

production-setup: check-requirements install train deploy test ## Complete production setup
	@echo "✅ Production setup completed"

# Monitoring and Maintenance
check-ollama: ## Check if Ollama is running
	@python -c "from utils import OllamaManager; mgr = OllamaManager(); print('Ollama Status:', '✓ Running' if mgr.is_ollama_running() else '✗ Not Running')"

list-models: ## List available Ollama models
	ollama list

model-info: ## Show model information
	@python -c "from utils import OllamaManager; mgr = OllamaManager('text-to-sql'); print(f'Model exists: {mgr.model_exists()}')"

# Benchmarking
benchmark: ## Run comprehensive benchmarks
	@echo "Running comprehensive benchmarks..."
	python test.py --config config.yaml --performance-only > benchmark_results.txt
	@echo "✅ Benchmark results saved to benchmark_results.txt"

compare-models: ## Compare different model configurations
	@echo "Comparing model configurations..."
	python text_to_sql_train.py --mode train --base-model "defog/sqlcoder-7b" --max-samples 1000
	python deploy.py --config config.yaml --test > sqlcoder_results.json
	python text_to_sql_train.py --mode train --base-model "codellama/CodeLlama-7b-Instruct-hf" --max-samples 1000
	python deploy.py --config config.yaml --test > codellama_results.json
	@echo "✅ Model comparison completed"

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	python -c "import pydoc; pydoc.writedoc('text_to_sql_train'); pydoc.writedoc('utils'); pydoc.writedoc('deploy'); pydoc.writedoc('test')"
	@echo "✅ Documentation generated"

# Validation
validate-config: ## Validate configuration file
	@python -c "from utils import ConfigManager; config = ConfigManager(); print('✅ Configuration is valid')"

validate-sql: ## Validate generated SQL samples
	@python -c "from utils import SQLValidator; from test import TextToSQLTester; tester = TextToSQLTester(); results = tester.run_comprehensive_test_suite(); print(f'SQL Validation: {results[\"overall_metrics\"][\"success_rate\"]:.1%} success rate')"

# Advanced Operations
export-model: ## Export trained model for distribution
	@echo "Exporting model..."
	mkdir -p exports/
	cp -r models/text-to-sql-final/ exports/text-to-sql-$(shell date +%Y%m%d)/
	cp config.yaml exports/text-to-sql-$(shell date +%Y%m%d)/
	tar -czf exports/text-to-sql-$(shell date +%Y%m%d).tar.gz -C exports/ text-to-sql-$(shell date +%Y%m%d)/
	@echo "✅ Model exported to exports/"

backup: ## Backup models and configuration
	@echo "Creating backup..."
	mkdir -p backups/
	tar -czf backups/backup-$(shell date +%Y%m%d-%H%M%S).tar.gz models/ config.yaml processed_data/
	@echo "✅ Backup created in backups/"

restore: ## Restore from backup (specify BACKUP_FILE)
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Error: Please specify BACKUP_FILE=<filename>"; exit 1; fi
	tar -xzf backups/$(BACKUP_FILE)
	@echo "✅ Restored from $(BACKUP_FILE)"

# CI/CD Helpers
ci-test: lint format-check test ## Run CI tests
	@echo "✅ CI tests completed"

pre-commit: format lint test-quick ## Pre-commit checks
	@echo "✅ Pre-commit checks passed"

# Health Checks
health-check: ## Comprehensive health check
	@echo "Running health checks..."
	@make check-requirements
	@make check-ollama
	@make validate-config
	@echo "✅ Health check completed"

# Examples and Demos
run-examples: ## Run example queries
	@echo "Running example queries..."
	python -c "
import subprocess
examples = [
    'CREATE TABLE users (id INT, name VARCHAR(50)); Find all users',
    'CREATE TABLE orders (id INT, total DECIMAL(10,2)); Calculate total revenue'
]
for example in examples:
    print(f'Example: {example}')
    result = subprocess.run(['ollama', 'run', 'text-to-sql', example], capture_output=True, text=True)
    print(f'Result: {result.stdout}')
    print('-' * 50)
"

# Version Management
version: ## Show current version
	@python -c "import json; print(json.load(open('package.json' if os.path.exists('package.json') else 'setup.py'))['version'] if 'version' in json.load(open('package.json' if os.path.exists('package.json') else 'setup.py')) else 'Version not found')" 2>/dev/null || echo "1.0.0"

# All-in-one commands
dev-all: dev-setup train-quick deploy-test ## Complete development workflow
	@echo "✅ Complete development workflow finished"

prod-all: production-setup backup ## Complete production workflow
	@echo "✅ Complete production workflow finished"

demo-all: train-quick deploy demo ## Quick demo setup
	@echo "✅ Demo setup completed"

# Help for specific areas
help-train: ## Show training-related commands
	@echo "Training Commands:"
	@echo "=================="
	@grep -E '^train.*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

help-deploy: ## Show deployment-related commands
	@echo "Deployment Commands:"
	@echo "==================="
	@grep -E '^deploy.*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

help-test: ## Show testing-related commands
	@echo "Testing Commands:"
	@echo "================"
	@grep -E '^test.*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

help-docker: ## Show Docker-related commands
	@echo "Docker Commands:"
	@echo "==============="
	@grep -E '^docker.*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
