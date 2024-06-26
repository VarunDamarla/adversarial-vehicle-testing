install:
	poetry install --with dev

fix:
	poetry run ruff check src --fix

lint:
	poetry run ruff check src

type:
	poetry run mypy src

test:
	poetry run pytest tests

check: lint type test

clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache

.PHONY: check clean install lint tests
