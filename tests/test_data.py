from src.data import load_data


def test_data_shapes():
    X_train, X_test, y_train, y_test = load_data()

    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13

    assert len(X_train) + len(X_test) == 178
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_target_classes():
    X_train, X_test, y_train, y_test = load_data()

    classes = set(y_train) | set(y_test)

    assert classes == {0, 1, 2}