from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained classification model.
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return accuracy