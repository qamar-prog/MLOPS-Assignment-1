import argparse

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.data import load_data


def train_model(n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier and log the run to MLflow.
    """
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            model,
            "model",
        )

        print(f"Test Accuracy: {accuracy:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    train_model(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()