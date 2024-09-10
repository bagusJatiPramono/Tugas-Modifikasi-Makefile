VENV := venv

PYTHON := $(VENV)\Scripts\python.exe

all: run-script prep train evaluate deploy

$(VENV)\Scripts\activate: requirements.txt
	python -m venv $(VENV)
	$(VENV)\Scripts\python.exe -m pip install --upgrade pip
	$(VENV)\Scripts\python.exe -m pip install -r requirements.txt

install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

run-script: install
	$(VENV)\Scripts\activate

prep:
	python scripts/data_prep.py

train:
	python scripts/train_model.py

evaluate:
	python scripts/evaluate_model.py

deploy:
	python scripts/deploy_model.py

