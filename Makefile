.PHONY: help install data clean test notebooks compare all

help:
	@echo "Volatility Forecasting: GARCH vs LSTM"
	@echo "======================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compare        - Run GARCH vs LSTM comparison (recommended)"
	@echo "  make notebooks      - Launch Jupyter notebooks"
	@echo ""
	@echo "Pipeline commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make test           - Run tests"
	@echo "  make clean          - Clean generated files"
	@echo "  make all            - Run full comparison"

install:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

compare:
	@echo "Running GARCH vs LSTM comparison..."
	python compare_garch_lstm.py SPY 2015-01-01 2024-10-28
	@echo ""
	@echo "✓ Comparison complete! Check the generated plots:"
	@echo "  - comparison_forecasts.png"
	@echo "  - comparison_errors.png"
	@echo "  - comparison_scatter.png"
	@echo "  - comparison_backtests.png"

garch-only:
	@echo "Running GARCH research only..."
	python run_garch_research.py SPY 2015-01-01 2024-10-28
	@echo "✓ GARCH research complete!"

notebooks:
	@echo "Launching Jupyter notebooks..."
	jupyter notebook notebooks/

test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short
	@echo "✓ Tests complete"

clean:
	@echo "Cleaning generated files..."
	rm -rf data/processed/*.parquet
	rm -rf models/*.pkl
	rm -rf results/*.png results/*.html
	rm -rf __pycache__ src/__pycache__ src/*/__pycache__ src/*/*/__pycache__
	rm -f *.png *.html
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned"

all: install compare
	@echo ""
	@echo "===================="
	@echo "✓ Pipeline complete!"
	@echo "===================="