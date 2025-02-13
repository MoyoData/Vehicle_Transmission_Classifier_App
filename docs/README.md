# Vehicle Transmission Classifier Makefile Guide

This project provides a Makefile to manage the environment setup and tasks for training and prediction of a vehicle transmission classification model.

## Makefile Commands

### Environment Setup
- **Setup Virtual Environment**

```bash
make init
```
This command sets up a Python virtual environment and installs dependencies from `requirements.txt`.

### Running Tasks
- **Preprocess Data**
```bash
make preprocess
```
Executes the `preprocess.py` script in the `src` directory to prepare the dataset.

- **Train the Model**
```bash
make train
```
Executes the `train.py` script located in the `src` directory to train the vehicle transmission classification model.

- **Run Predictions**
```bash
make predict
```
Executes the `predict.py` script located in the `src` directory to make predictions with the trained model.

- **Evaluate the Model**
```bash
make evaluate
```
Executes the `evaluate.py` script located in the `src` directory to evaluate model performance.

### Cleaning Up
- **Clean Cache and Logs**
```bash
make clean
```
Removes `__pycache__` directories, the virtual environment, and cache files.

### Help
- **Display Available Commands**
```bash
make help
```
Lists all available Makefile commands with a brief description.

---

## Instructions for Environment Setup

1. Use `make init` to set up the environment.
2. Activate the virtual environment:
```bash
source .venv/bin/activate
```
3. Install additional dependencies by updating `requirements.txt`.

---

## Notes
- Ensure you have Python 3.8+ installed.
- The `clean` command will remove the virtual environment, so use it with caution.


