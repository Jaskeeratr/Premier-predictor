.PHONY: install install-dev run test lint format docker-up frontend-build clean-generated ci-check

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

clean-generated:
	python -c "import pathlib,shutil; root=pathlib.Path('.'); [p.unlink(missing_ok=True) for p in root.rglob('*.pyc')]; [shutil.rmtree(p, ignore_errors=True) for p in root.rglob('__pycache__')]; [p.unlink(missing_ok=True) for p in root.glob('*.db')]; [p.unlink(missing_ok=True) for p in root.rglob('*.sqlite')]; [p.unlink(missing_ok=True) for p in root.rglob('*.sqlite3')]"
	python -c "import shutil,pathlib; [shutil.rmtree(path, ignore_errors=True) for path in (pathlib.Path('.pytest_cache'), pathlib.Path('.ruff_cache'), pathlib.Path('artifacts')) if path.exists()]"

ci-check:
	ruff check .
	black --required-version 26.5.1 --check .
	python -m pytest
	npm run check:frontend
	npm run build:frontend
