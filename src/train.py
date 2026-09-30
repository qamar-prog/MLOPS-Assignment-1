import mlflow
import mlflow.sklearn
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import load_wine_data


# ---------------------------------------------------------
# Hyperparameter search grids
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Model creation
# ---------------------------------------------------------

def create_random_forest(params):
    """
    Create a Random Forest classifier.
    """
    return RandomForestClassifier(
        **params,
        random_state=42,
    )


def create_gradient_boosting(params):
    """
    Create a Gradient Boosting classifier.
    """
    return GradientBoostingClassifier(
        **params,
        random_state=42,
    )


# ---------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------

def evaluate_configuration(model, X_train, y_train):
    """
    Evaluate one model configuration using 5-fold
    stratified cross-validation.
    """
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

    metrics = {
        "train_accuracy": results["train_accuracy"].mean(),
        "validation_accuracy": results["test_accuracy"].mean(),
        "train_f1_macro": results["train_f1_macro"].mean(),
        "validation_f1_macro": results["test_f1_macro"].mean(),
        "train_log_loss": -results["train_log_loss"].mean(),
        "validation_log_loss": -results["test_log_loss"].mean(),
    }

    return metrics


# ---------------------------------------------------------
# Random Forest experiment
# ---------------------------------------------------------

def train_random_forest(X_train, y_train):
    """
    Run all Random Forest configurations and log them to MLflow.
    """
    best_model = None
    best_score = -1
    best_params = None

    for config_number, params in enumerate(
        RANDOM_FOREST_GRID,
        start=1,
    ):
        model = create_random_forest(params)

        with mlflow.start_run(
            run_name=f"RandomForest_Config_{config_number}"
        ):
            metrics = evaluate_configuration(
                model,
                X_train,
                y_train,
            )

            mlflow.log_param("model_family", "RandomForest")
            mlflow.log_param("config_number", config_number)

            for name, value in params.items():
                mlflow.log_param(name, value)

            mlflow.log_metrics(metrics)

            model.fit(X_train, y_train)

            mlflow.sklearn.log_model(
                model,
                "model",
            )

            print(
                f"\nRandom Forest - Configuration "
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

            if metrics["validation_f1_macro"] > best_score:
                best_score = metrics["validation_f1_macro"]
                best_model = model
                best_params = params

    print("\nBest Random Forest configuration:")
    print(best_params)

    return best_model, best_params


# ---------------------------------------------------------
# Gradient Boosting experiment
# ---------------------------------------------------------

def train_gradient_boosting(X_train, y_train):
    """
    Run all Gradient Boosting configurations and log them
    to MLflow.
    """
    best_model = None
    best_score = -1
    best_params = None

    for config_number, params in enumerate(
        GRADIENT_BOOSTING_GRID,
        start=1,
    ):
        model = create_gradient_boosting(params)

        with mlflow.start_run(
            run_name=f"GradientBoosting_Config_{config_number}"
        ):
            metrics = evaluate_configuration(
                model,
                X_train,
                y_train,
            )

            mlflow.log_param(
                "model_family",
                "GradientBoosting",
            )
            mlflow.log_param(
                "config_number",
                config_number,
            )

            for name, value in params.items():
                mlflow.log_param(name, value)

            mlflow.log_metrics(metrics)

            model.fit(X_train, y_train)

            mlflow.sklearn.log_model(
                model,
                "model",
            )

            print(
                f"\nGradient Boosting - Configuration "
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

            if metrics["validation_f1_macro"] > best_score:
                best_score = metrics["validation_f1_macro"]
                best_model = model
                best_params = params

    print("\nBest Gradient Boosting configuration:")
    print(best_params)

    return best_model, best_params


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main():
    """
    Run the complete dual-classifier training pipeline.
    """
    mlflow.set_experiment("Wine_Classifier_Experiments")

    X_train, X_test, y_train, y_test = load_wine_data()

    print("Wine dataset loaded successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")

    print("\n" + "=" * 60)
    print("RANDOM FOREST EXPERIMENTS")
    print("=" * 60)

    train_random_forest(
        X_train,
        y_train,
    )

    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING EXPERIMENTS")
    print("=" * 60)

    train_gradient_boosting(
        X_train,
        y_train,
    )

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()