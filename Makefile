.PHONY: install data demo guardrail evals test api graph n8n-workflow docker docker-email

install:
	pip install -r requirements.txt

data:
	python -c "import sys; sys.path.insert(0,'src'); from gridwise.data import build_database; print(build_database())"

demo:
	python scripts/run_demo.py

guardrail:
	python scripts/run_guardrail_demo.py

evals:
	python scripts/run_evals.py --report data/eval_report.json

test:
	python -m pytest

api:
	PYTHONPATH=src uvicorn gridwise.api:app --reload --port 8000

graph:
	python scripts/show_graph.py

n8n-workflow:
	python scripts/build_n8n_workflow.py

docker:
	docker compose up --build

docker-email:
	docker compose --profile email up --build
