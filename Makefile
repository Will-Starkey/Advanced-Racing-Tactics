.PHONY: install test test-cov run lint

install:
	python3 -m venv venv
	venv/bin/pip install -r requirements-dev.txt

test:
	venv/bin/pytest

test-cov:
	venv/bin/pytest --cov=. --cov-report=term-missing --cov-omit="venv/*,tests/*"

run:
	source venv/bin/activate && python main.py

lint:
	venv/bin/python -m py_compile main.py tactics_engine.py llm_bridge.py signalk_client.py \
		polar/parser.py polar/models.py polar/manager.py \
		instruments/base.py instruments/bg_adapter.py instruments/garmin_adapter.py
	@echo "Syntax OK"
