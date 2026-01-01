.PHONY: help install train tune run clean mlflow

# Virtual Environment config
VENV_NAME := venv
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip

help:
	@echo "Available commands:"
	@echo "  make venv     - Create virtual environment"
	@echo "  make install  - Install dependencies in venv"
	@echo "  make train    - Train the model (usage: make train ARGS=\"--epochs 10\")"
	@echo "  make tune     - Run hyperparameter tuning (usage: make tune ARGS=\"--trials 20\")"
	@echo "  make run      - Run the Streamlit app"
	@echo "  make mlflow   - Run MLflow UI"
	@echo "  make clean    - Remove temporary files"

# Create venv if it doesn't exist
$(VENV_NAME)/bin/activate: requirements.txt
	test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	touch $(VENV_NAME)/bin/activate

venv: $(VENV_NAME)/bin/activate

install: venv
	$(PIP) install -r requirements.txt

# Use ARGS to pass arguments to scripts
# e.g. make train ARGS="--epochs 5"
train: venv
	$(PYTHON) src/models/train_model.py $(ARGS)

tune: venv
	$(PYTHON) src/models/tune.py $(ARGS)

run: venv
	$(VENV_NAME)/bin/streamlit run app.py

mlflow: venv
	$(VENV_NAME)/bin/mlflow ui --port 5555

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
