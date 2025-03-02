.PHONY: install dev test lint format clean dashboard run-react run-plan

install:
	pip install -e .

dev:
	pip install -e ".[dev,dashboard]"

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

dashboard:
	uvicorn agent_framework.dashboard.app:app --host 0.0.0.0 --port 8080 --reload

run-react:
	python examples/research_assistant.py

run-plan:
	python examples/math_solver.py
