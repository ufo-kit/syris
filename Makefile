PYTHON = python
SETUP = $(PYTHON) setup.py

.PHONY: build clean check check-fast dist init install

all: build

install: clean build
	$(SETUP) install

build:
	$(SETUP) build

dist:
	$(SETUP) sdist

clean:
	$(SETUP) clean --all

init:
	pip install -r ./requirements.txt
