# Vehicle Transmission Classifier Makefile

# Project Variables
VENV=.vehicle_classifier
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

# Create virtual environment for CPU usage
init:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete! Run 'source $(VENV)/bin/activate' to activate the environment."

preprocess:
	@echo "Running data preprocessing..."
	$(PYTHON) src/preprocess.py

train:
	@echo "Training the vehicle transmission model..."
	$(PYTHON) src/train.py

predict:
	@echo "Running predictions..."
	$(PYTHON) src/predict.py

evaluate:
	@echo "Evaluating the model..."
	$(PYTHON) src/evaluate.py

clean:
	@echo "Cleaning up cache files and virtual environment..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf $(VENV)
	rm -rf .pytest_cache

help:
	@echo "Available Makefile commands:"
	@echo "  make init       - Create virtual environment and install dependencies"
	@echo "  make preprocess - Run data preprocessing script"
	@echo "  make train      - Train the model"
	@echo "  make predict    - Run predictions using the trained model"
	@echo "  make evaluate   - Evaluate the trained model"
	@echo "  make clean      - Remove cache files and virtual environment"


