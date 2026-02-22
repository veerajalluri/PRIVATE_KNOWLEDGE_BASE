# Any args passed to the make script, use with $(call args, default_value)
args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`

########################################################################################################################
# Quality checks
########################################################################################################################

test:
	PYTHONPATH=. poetry run pytest tests

test-coverage:
	PYTHONPATH=. poetry run pytest tests --cov private_gpt --cov-report term --cov-report=html --cov-report xml --junit-xml=tests-results.xml

black:
	poetry run black . --check

ruff:
	poetry run ruff check private_gpt tests

format:
	poetry run black .
	poetry run ruff check private_gpt tests --fix

mypy:
	poetry run mypy private_gpt

check:
	make format
	make mypy

########################################################################################################################
# Run
########################################################################################################################

run:
	PGPT_PROFILES=local poetry run python -m private_gpt

dev:
	PYTHONPATH=. PYTHONUNBUFFERED=1 PGPT_PROFILES=local poetry run python -m uvicorn private_gpt.main:app --reload --port 8001

########################################################################################################################
# BI data pipeline
########################################################################################################################

prepare:
	poetry run python scripts/prepare_bi_data.py data/uploads data/aggregated

ingest:
	PGPT_PROFILES=local poetry run python scripts/ingest_bi_json.py data/aggregated

bi: prepare ingest
	@echo "Data pipeline complete. Run 'make run' to start the app."

########################################################################################################################
# Docker
########################################################################################################################

docker-build:
	docker compose -f docker-compose.bi.yaml build

docker-up: _dirs
	docker compose -f docker-compose.bi.yaml up

# Ensure host-side mount points exist before Docker creates them as root-owned dirs
_dirs:
	@mkdir -p data/uploads data/aggregated local_data models

docker-down:
	docker compose -f docker-compose.bi.yaml down

docker-logs:
	docker compose -f docker-compose.bi.yaml logs -f

########################################################################################################################
# Misc
########################################################################################################################

wipe:
	poetry run python scripts/utils.py wipe

setup:
	poetry run python scripts/setup

list:
	@echo "Quality checks:"
	@echo "  test            Run tests with pytest"
	@echo "  test-coverage   Run tests with coverage report"
	@echo "  format          Format code with black and ruff"
	@echo "  mypy            Type-check with mypy"
	@echo "  check           Format + mypy"
	@echo ""
	@echo "Run:"
	@echo "  run             Start the app (local profile)"
	@echo "  dev             Start with auto-reload for development"
	@echo ""
	@echo "BI data pipeline:"
	@echo "  prepare         Aggregate raw JSON in data/uploads/ → data/aggregated/"
	@echo "  ingest          Embed aggregated summaries into Qdrant"
	@echo "  bi              prepare + ingest (full pipeline)"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build the conv-bi Docker image"
	@echo "  docker-up       Start the full stack (Ollama + conv-bi)"
	@echo "  docker-down     Stop the stack"
	@echo "  docker-logs     Tail container logs"
	@echo ""
	@echo "Misc:"
	@echo "  wipe            Wipe the vector store and docstore"
	@echo "  setup           Download models (llamacpp mode only)"
