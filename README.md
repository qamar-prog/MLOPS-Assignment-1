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