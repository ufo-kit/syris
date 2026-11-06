UV ?= uv
CUDA_EXTRA ?= cuda13

.PHONY: all build dist install init lint format check check-cuda docs

all: build

build dist:
	$(UV) build

install init:
	$(UV) sync

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff check --select I --fix .
	$(UV) run ruff format .

check:
	$(UV) run python -m unittest discover -v

check-cuda:
	SYRIS_REQUIRE_CUDA=1 $(UV) run --extra $(CUDA_EXTRA) python -m unittest \
		syris.tests.test_cuda_projection syris.tests.test_cuda_scheduling -v

docs:
	$(UV) run --group docs sphinx-build -b html docs/source docs/build/html
