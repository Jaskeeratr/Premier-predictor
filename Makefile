.PHONY: install install-dev run test lint format docker-up frontend-build

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

run:
	python run_webapp.py

test:
	pytest

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check . --fix

docker-up:
	docker compose up --build

frontend-build:
	npm ci
	npm run build:frontend
