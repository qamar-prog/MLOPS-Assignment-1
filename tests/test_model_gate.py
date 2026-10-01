import time
import mlflow
import numpy as np
from sklearn.metrics import f1_score
from src.data import load_wine_data

MODEL_URI = "models:/WineClassifier@champion"


def load_champion_model():
    """
    Load the production champion model from MLflow registry.
    """
    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    return model


def test_validation_macro_f1_gate():

    """
    Quality Gate:
    Validation Macro F1 must be >= 0.88
    """

    model = load_champion_model()

    X_train, X_test, y_train, y_test = (
        load_wine_data()
    )

    predictions = model.predict(
        X_test
    )

    score = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    print(
        f"Validation Macro F1: {score}"
    )

    assert score >= 0.88, (
        f"Model failed F1 gate: {score}"
    )


def test_inference_latency_gate():

    """
    Quality Gate:
    Batch inference must be <= 30ms
    """

    model = load_champion_model()

    X_train, X_test, y_train, y_test = (
        load_wine_data()
    )

    start_time = time.time()

    model.predict(
        X_test
    )

    end_time = time.time()

    latency_ms = (
        end_time - start_time
    ) * 1000

    print(
        f"Inference latency: {latency_ms:.4f} ms"
    )

    assert latency_ms <= 30, (
        f"Latency too high: {latency_ms}"
    )


def test_output_schema_integrity():

    """
    Quality Gate:
    Predictions must contain only
    valid class indices.
    """

    model = load_champion_model()

    X_train, X_test, y_train, y_test = (
        load_wine_data()
    )

    predictions = model.predict(
        X_test
    )

    allowed_classes = {
        0,
        1,
        2,
    }

    assert set(
        np.unique(predictions)
    ).issubset(
        allowed_classes
    )

