PYTHON ?= python

.PHONY: test test-generation test-evaluation smoke-generation

test: test-generation test-evaluation

test-generation:
	cd poster_generation && PYTHONPATH=. $(PYTHON) -m pytest -q

test-evaluation:
	cd benchmark_eval && $(PYTHON) -m unittest discover -s tests -v
	cd benchmark_eval && $(PYTHON) -m compileall -q common prr_che universal_score keypoint_bertscore

smoke-generation:
	cd poster_generation && PYTHONPATH=. $(PYTHON) -m src.workflow.pipeline --list-layout-templates
