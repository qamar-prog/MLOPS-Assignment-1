import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)

from src.data import load_wine_data


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_URI = (
    "models:/WineClassifier@champion"
)


def evaluate_champion():
    """
    Load the registered champion model and evaluate it
    on the held-out test split.
    """
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    X_train, X_test, y_train, y_test = (
        load_wine_data()
    )

    print("Loading champion model...")

    model = mlflow.sklearn.load_model(
        REGISTERED_MODEL_URI
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    test_log_loss = log_loss(
        y_test,
        probabilities,
    )

    print("\n" + "=" * 60)
    print("CHAMPION MODEL TEST EVALUATION")
    print("=" * 60)

    print(
        f"Test Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"Test Log Loss: "
        f"{test_log_loss:.4f}"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "log_loss": test_log_loss,
    }


if __name__ == "__main__":
    evaluate_champion()
