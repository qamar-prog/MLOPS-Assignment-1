# Wine MLOps Pipeline

An automated MLOps pipeline for multi-class chemical cultivar classification
using the Scikit-learn Wine dataset.

## Dataset

The project uses `sklearn.datasets.load_wine`.

- Samples: 178
- Features: 13
- Classes: 3

## Project Structure

```text
wine-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
├── src/
│   ├── data.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── test_data.py
│   └── test_model_gate.py
├── .gitignore
├── Makefile
├── requirements.txt
└── README.md

## Milestones

### Milestone 1: Local Automation and Environment Management

The project was initialized with a reproducible Python environment and local automation.

Implemented:

* Python virtual environment
* Pinned project dependencies
* Makefile automation
* Flake8 linting
* Pytest unit testing
* Git version control
* `.gitignore` for generated files and environments

Available commands:

```bash
make install
make lint
make test
make train
make clean
```

---

### Milestone 2: Modular Pipeline and Dual Classifier Training

The training pipeline was extended to support multiple classifier families and hyperparameter configurations.

Implemented:

* Stratified 80/20 train-test split
* `random_state=42`
* Dataset validation
* Random Forest classifier
* Gradient Boosting classifier
* Three configurations for each classifier family
* 5-fold Stratified Cross-Validation
* Accuracy
* Macro F1-score
* Log Loss
* MLflow experiment tracking

The pipeline evaluates six candidate model configurations in total.

---

### Milestone 3: Comprehensive MLflow Tracking and Model Registry

MLflow was integrated for experiment tracking and model management.

Implemented:

* Local SQLite MLflow backend
* Experiment: `Wine-Cultivar-Classification`
* Individual MLflow run for every candidate configuration
* Parameter logging
* Training and validation metric logging
* Model artifact logging
* MLflow model signatures
* Training input examples
* Model Registry
* Registered model: `WineClassifier`
* Champion alias: `champion`

The best candidate is selected using validation Macro F1-score and registered as the champion model.

The MLflow UI can be started with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

---

### Milestone 4: CI/CD Pipeline and Automated MLOps Quality Gate

GitHub Actions was configured to automatically validate the project on code changes.

The CI workflow runs on:

* Push to `main`
* Pull requests targeting `main`

The workflow uses:

* Ubuntu latest
* Python 3.10
* `make install`
* `make lint`
* `make test`

The automated model quality gate verifies:

#### Metric Threshold Gate

Validation Macro F1-score must be at least:

```text
0.88
```

#### Inference Latency Gate

Batch inference must complete within:

```text
30 ms
```

#### Output Schema Integrity

Model predictions must contain only valid Wine class indices:

```text
0, 1, 2
```

A failure in any quality gate causes the CI workflow to fail.

---

### Milestone 5: Git Collaboration and Conflict Resolution

A feature-branch workflow was used to demonstrate distributed version control and merge conflict resolution.

A dedicated conflict simulation branch was created:

```bash
git switch -c conflict-simulation
```

The same configuration line in `Makefile` was intentionally modified differently on the feature branch and `main`.

The feature branch contained:

```text
flake8 src/ tests/ --max-line-length=120
```

The `main` branch contained:

```text
flake8 src/ tests/ --max-line-length=80
```

The feature branch was then merged into `main`:

```bash
git switch main
git merge conflict-simulation
```

Git detected a merge conflict in `Makefile`.

The conflict markers were manually removed and the final configuration was resolved to:

```text
flake8 src/ tests/ --max-line-length=100
```

The merge was completed with:

```bash
git add Makefile
git commit -m "Resolve merge conflict in lint configuration"
```

The merge history was verified using:

```bash
git log --oneline --graph --all
```

Example resulting history:

```text
*   a83119d Resolve merge conflict in lint configuration
|\
| * 845eb5c Configure lint line length for conflict simulation
* | 4eb1514 Adjust lint configuration on main
|/
* 8c3914b Milestone 4: CI/CD Pipeline and Automated MLOps Quality Gate
```

This demonstrates:

* Feature branching
* Independent branch development
* Intentional merge conflict creation
* Manual conflict resolution
* Merge commit creation
* Git history verification

---

## How to Run

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
make install
```

### 3. Run Linting

```bash
make lint
```

### 4. Run Unit Tests

```bash
make test
```

### 5. Train Models

```bash
make train
```

This trains all configured candidate models and records their experiments in MLflow.

### 6. Evaluate Champion Model

```bash
make evaluate
```

This loads the registered `WineClassifier@champion` model and evaluates it on the held-out test set.

### 7. Open MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```text
http://127.0.0.1:5000
```

---

## MLOps Workflow

```text
Dataset
   │
   ▼
Data Validation
   │
   ▼
Stratified Train/Test Split
   │
   ▼
Candidate Model Training
   │
   ├── Random Forest
   │     ├── Configuration 1
   │     ├── Configuration 2
   │     └── Configuration 3
   │
   └── Gradient Boosting
         ├── Configuration 1
         ├── Configuration 2
         └── Configuration 3
   │
   ▼
5-Fold Cross-Validation
   │
   ▼
MLflow Tracking
   │
   ▼
Best Validation Macro F1
   │
   ▼
WineClassifier Registry
   │
   ▼
champion Alias
   │
   ▼
Quality Gates
   ├── Macro F1 ≥ 0.88
   ├── Latency ≤ 30 ms
   └── Valid Classes {0, 1, 2}
   │
   ▼
GitHub Actions CI
```

---

## Technologies Used

* Python 3.10
* Scikit-learn
* MLflow
* Pandas
* NumPy
* Pytest
* Flake8
* Make
* Git
* GitHub Actions

## Quality Assurance

The project uses automated checks to maintain code and model quality:

```text
Code Change
    ↓
GitHub Actions
    ↓
Install Dependencies
    ↓
Lint
    ↓
Unit Tests
    ↓
MLOps Quality Gates
    ↓
Pipeline Validation
```

A change that fails the required checks is prevented from passing the CI validation workflow.

## License

This project is developed for academic and educational purposes as part of an MLOps pipeline assignment.
