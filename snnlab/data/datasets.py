from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_iris_dataset(*, seed: int = 52) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and min-max normalizes Iris with a stratified train/test split.

    Загружает Iris, выполняет min-max нормализацию и стратифицированное разбиение.
    """
    iris = load_iris()
    x = MinMaxScaler().fit_transform(iris.data).astype(np.float64)
    y = iris.target.astype(np.int64)
    return train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)


def load_mnist_dataset(
    *,
    train_limit: int | None = None,
    test_limit: int | None = None,
    subset_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads MNIST and optionally reproduces the notebook's shuffled subset protocol.

    When ``subset_seed`` is provided, train and test indices are shuffled by one
    shared ``numpy.random.Generator`` before applying ``train_limit`` and
    ``test_limit``. This exactly matches the current DCI notebook data-pool
    construction and is intentionally separate from the later training schedule.

    Загружает MNIST и при необходимости воспроизводит протокол shuffled subset
    из блокнота.

    Если передан ``subset_seed``, train- и test-индексы перемешиваются одним
    общим ``numpy.random.Generator`` до применения ``train_limit`` и
    ``test_limit``. Это точно соответствует формированию data pool в текущем
    DCI-блокноте и намеренно отделено от последующего расписания обучения.
    """
    try:
        from tensorflow.keras.datasets import mnist
    except ImportError as exc:
        raise ImportError(
            "MNIST loader requires the optional dependency: pip install -e '.[mnist]'"
        ) from exc

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float64) / 255.0
    y_train = y_train.astype(np.int64)
    x_test = x_test.astype(np.float64) / 255.0
    y_test = y_test.astype(np.int64)

    if subset_seed is not None:
        rng = np.random.default_rng(subset_seed)
        train_indices = np.arange(len(x_train), dtype=np.int64)
        test_indices = np.arange(len(x_test), dtype=np.int64)
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

        if train_limit is not None:
            train_indices = train_indices[:train_limit]
        if test_limit is not None:
            test_indices = test_indices[:test_limit]

        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
        x_test = x_test[test_indices]
        y_test = y_test[test_indices]
    else:
        if train_limit is not None:
            x_train = x_train[:train_limit]
            y_train = y_train[:train_limit]
        if test_limit is not None:
            x_test = x_test[:test_limit]
            y_test = y_test[:test_limit]

    return x_train, y_train, x_test, y_test
