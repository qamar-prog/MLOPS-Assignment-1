import numpy as np

from src.data import load_wine_data


def test_data_shapes():
    X_train, X_test, y_train, y_test = load_wine_data()

    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13

    assert len(X_train) + len(X_test) == 178
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_target_classes():
    X_train, X_test, y_train, y_test = load_wine_data()

    classes = set(y_train) | set(y_test)

    assert classes == {0, 1, 2}


def test_no_null_values():
    X_train, X_test, y_train, y_test = load_wine_data()

    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert not np.isnan(y_train).any()
    assert not np.isnan(y_test).any()


def test_feature_count():
    X_train, X_test, _, _ = load_wine_data()

    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13
