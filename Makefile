.PHONY: data serve eval test

data:
	python -m app.setup

serve:
	python -m uvicorn app.main:app --port 8041

eval:
	python -m eval.run_eval
	python -m eval.run_eval eval/blind.jsonl eval/results-blind.json
	python -m eval.run_eval eval/paraphrase.jsonl eval/results-paraphrase.json
	python -m eval.charts

test:
	python -m pytest -q
