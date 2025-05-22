.PHONY: install clean format check-format
# Makefile for Python project
install:
	@uv pip install -r requirements.txt

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "Cleaned up generated files."

format:
	@black .
	@echo "Formatted code with Black."

check-format:
	@black --check .
	@echo "Checked code format with Black."

