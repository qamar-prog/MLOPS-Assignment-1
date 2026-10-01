import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import load_wine_data


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Wine-Cultivar-Classification"
REGISTERED_MODEL_NAME = "WineClassifier"


RANDOM_FOREST_GRID = [
    {
        "n_estimators": 50,
        "max_depth": 3,
        "min_samples_split": 2,
    },
    {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
    },
]


GRADIENT_BOOSTING_GRID = [
    {
        "n_estimators": 50,
        "learning_rate": 0.05,
        "max_depth": 2,
    },
    {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
    },
    {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 4,
    },
]


def create_random_forest(params):
    """Create a Random Forest classifier."""
    return RandomForestClassifier(
        **params,
        random_state=42,
    )


def create_gradient_boosting(params):
    """Create a Gradient Boosting classifier."""
    return GradientBoostingClassifier(
        **params,
        random_state=42,
    )


def evaluate_configuration(model, X_train, y_train):
    """Evaluate one configuration using 5-fold stratified CV."""
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "log_loss": "neg_log_loss",
    }

    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    return {
        "train_accuracy": results["train_accuracy"].mean(),
        "validation_accuracy": results["test_accuracy"].mean(),
        "train_f1_macro": results["train_f1_macro"].mean(),
        "validation_f1_macro": results["test_f1_macro"].mean(),
        "train_log_loss": -results["train_log_loss"].mean(),
        "validation_log_loss": -results["test_log_loss"].mean(),
    }


def run_experiment(
    model_family,
    config_number,
    params,
    model,
    X_train,
    y_train,
):
    """
    Train one candidate configuration and log the complete
    experiment to MLflow.
    """
    metrics = evaluate_configuration(
        model,
        X_train,
        y_train,
    )

    # Fit final candidate on the complete training split.
    model.fit(X_train, y_train)

    # Generate predictions for signature inference.
    training_predictions = model.predict(X_train)

    signature = infer_signature(
        X_train,
        training_predictions,
    )

    # Save a small example from the training split.
    input_example = X_train[:1]

    with mlflow.start_run(
        run_name=f"{model_family}_Config_{config_number}"
    ) as run:

        # -----------------------------
        # Tags
        # -----------------------------
        mlflow.set_tag(
            "model_family",
            model_family,
        )

        mlflow.set_tag(
            "stage",
            "candidate",
        )

        # -----------------------------
        # Parameters
        # -----------------------------
        mlflow.log_param(
            "config_number",
            config_number,
        )

        for name, value in params.items():
            mlflow.log_param(name, value)

        mlflow.log_param(
            "random_state",
            42,
        )

        mlflow.log_param(
            "cv_folds",
            5,
        )

        # -----------------------------
        # Metrics
        # -----------------------------
        mlflow.log_metrics(metrics)

        # -----------------------------
        # Model artifact
        # -----------------------------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        print(
            f"\n{model_family} - Configuration "
            f"{config_number}"
        )
        print(f"Parameters: {params}")

        print(
            f"Train Accuracy: "
            f"{metrics['train_accuracy']:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{metrics['validation_accuracy']:.4f}"
        )

        print(
            f"Train Macro F1: "
            f"{metrics['train_f1_macro']:.4f}"
        )

        print(
            f"Validation Macro F1: "
            f"{metrics['validation_f1_macro']:.4f}"
        )

        print(
            f"Train Log Loss: "
            f"{metrics['train_log_loss']:.4f}"
        )

        print(
            f"Validation Log Loss: "
            f"{metrics['validation_log_loss']:.4f}"
        )

        print(f"MLflow Run ID: {run.info.run_id}")

        return {
            "run_id": run.info.run_id,
            "model": model,
            "params": params,
            "metrics": metrics,
        }


def train_all_models(X_train, y_train):
    """Train every candidate configuration."""
    results = []

    for config_number, params in enumerate(
        RANDOM_FOREST_GRID,
        start=1,
    ):
        model = create_random_forest(params)

        result = run_experiment(
            "RandomForest",
            config_number,
            params,
            model,
            X_train,
            y_train,
        )

        results.append(result)

    for config_number, params in enumerate(
        GRADIENT_BOOSTING_GRID,
        start=1,
    ):
        model = create_gradient_boosting(params)

        result = run_experiment(
            "GradientBoosting",
            config_number,
            params,
            model,
            X_train,
            y_train,
        )

        results.append(result)

    return results


def promote_champion():
    """
    Find the candidate with the highest validation Macro F1,
    register its model, and assign the champion alias.
    """
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'candidate'",
        order_by=[
            "metrics.validation_f1_macro DESC"
        ],
    )

    if not runs:
        raise RuntimeError(
            "No candidate MLflow runs were found."
        )

    best_run = runs[0]

    best_run_id = best_run.info.run_id
    best_f1 = best_run.data.metrics[
        "validation_f1_macro"
    ]

    model_uri = f"runs:/{best_run_id}/model"

    print("\n" + "=" * 60)
    print("CHAMPION MODEL")
    print("=" * 60)

    print(f"Run ID: {best_run_id}")
    print(
        f"Validation Macro F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Model Family: "
        f"{best_run.data.tags.get('model_family')}"
    )

    print(
        f"Model URI: {model_uri}"
    )

    # Register the winning model.
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    version = registered_model.version

    # Assign champion alias.
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=version,
    )

    print(
        f"Registered Model: "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(f"Version: {version}")
    print("Alias: champion")

    return best_run_id, version


def main():
    """Run the complete MLflow training pipeline."""
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    X_train, X_test, y_train, y_test = (
        load_wine_data()
    )

    print("Wine dataset loaded successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(
        f"Number of features: "
        f"{X_train.shape[1]}"
    )

    print("\n" + "=" * 60)
    print("MLFLOW EXPERIMENTS")
    print("=" * 60)

    train_all_models(
        X_train,
        y_train,
    )

    promote_champion()

    print("\nTraining and model promotion completed.")


if __name__ == "__main__":
    main()
