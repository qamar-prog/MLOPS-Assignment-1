import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_wine_data(test_size=0.2, random_state=42):
    """
    Load the Wine dataset and perform a stratified train-test split.
    """
    wine = load_wine()

    X = wine.data
    y = wine.target

    validate_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def validate_data(X, y):
    """
    Validate the Wine dataset.
    """
    if np.isnan(X).any():
        raise ValueError("Dataset contains null values.")

    if np.isnan(y).any():
        raise ValueError("Target contains null values.")

    if X.shape[1] != 13:
        raise ValueError(
            f"Expected 13 features, but found {X.shape[1]}."
        )

    if len(X) != len(y):
        raise ValueError("Features and target have different lengths.")

    return True
