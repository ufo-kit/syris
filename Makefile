PYTHON = python
SETUP = $(PYTHON) setup.py
RUNTEST = nosetests syris

.PHONY: build clean check check-fast dist init install

all: build

install: clean build
	$(SETUP) install --record install_manifest.txt

uninstall:
	cat install_manifest.txt | xargs rm -rf

build:
	$(SETUP) build

dist:
	$(SETUP) sdist

clean:
	$(SETUP) clean --all

check:
	$(RUNTEST)

check-fast:
	$(RUNTEST) -a '!slow'

init:
	pip install -r ./requirements.txt
