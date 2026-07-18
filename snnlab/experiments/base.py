from __future__ import annotations

import numpy as np


def create_sample_schedule(
    *,
    n_items: int,
    n_samples: int,
    seed: int,
    shuffle: bool = True,
    allow_repeats: bool = False,
) -> np.ndarray:
    """
    Creates a reproducible sample schedule and stores repetition policy explicitly.

    Создаёт воспроизводимое расписание sample и явно учитывает политику повторов.
    """
    if n_items <= 0 or n_samples < 0:
        raise ValueError("n_items must be positive and n_samples non-negative")
    if n_samples == 0:
        return np.empty(0, dtype=np.int64)
    if not allow_repeats and n_samples > n_items:
        raise ValueError("n_samples exceeds n_items while allow_repeats=False")

    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    remaining = n_samples

    while remaining > 0:
        indices = np.arange(n_items, dtype=np.int64)
        if shuffle:
            rng.shuffle(indices)
        take = min(remaining, n_items)
        chunks.append(indices[:take])
        remaining -= take
        if remaining > 0 and not allow_repeats:
            raise RuntimeError("Internal schedule construction error")

    return np.concatenate(chunks)


def extend_sample_schedule(
    existing: np.ndarray,
    *,
    n_items: int,
    additional_samples: int,
    seed: int,
    shuffle: bool = True,
    allow_repeats: bool = True,
) -> np.ndarray:
    """
    Appends a reproducible continuation schedule without modifying processed indices.

    When allow_repeats=False, new indices are sampled only from dataset items that
    do not already occur in the existing schedule. This makes "train 300, then
    continue for 200 unique samples" an explicit and testable policy.

    Добавляет воспроизводимое расписание продолжения, не меняя уже обработанные индексы.

    При allow_repeats=False новые индексы выбираются только среди объектов, которых
    ещё нет в существующем расписании. Так сценарий «обучить 300, затем продолжить
    ещё 200 уникальных объектов» становится явной и проверяемой политикой.
    """
    existing = np.asarray(existing, dtype=np.int64)

    if additional_samples < 0:
        raise ValueError("additional_samples must be non-negative")
    if additional_samples == 0:
        return existing.copy()
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if np.any(existing < 0) or np.any(existing >= n_items):
        raise ValueError("existing schedule contains out-of-range indices")

    if allow_repeats:
        extra = create_sample_schedule(
            n_items=n_items,
            n_samples=additional_samples,
            seed=seed,
            shuffle=shuffle,
            allow_repeats=True,
        )
        return np.concatenate([existing, extra])

    used = np.unique(existing)
    available = np.setdiff1d(
        np.arange(n_items, dtype=np.int64),
        used,
        assume_unique=False,
    )
    if additional_samples > available.size:
        raise ValueError(
            "Not enough unused dataset items for allow_repeats=False: "
            f"requested {additional_samples}, available {available.size}"
        )

    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(available)
    extra = available[:additional_samples]
    return np.concatenate([existing, extra])


def create_epoch_schedule(
    *,
    n_items: int,
    samples_per_epoch: int,
    n_epochs: int,
    seed: int,
    shuffle_each_epoch: bool = True,
) -> np.ndarray:
    """
    Creates a deterministic multi-epoch schedule over one fixed training subset.

    The training subset is selected once from the available pool. Every epoch
    then traverses the same subset, optionally in a new deterministic order.
    This matches the usual meaning of an epoch: another pass over the same
    selected training data rather than silent resampling from the full pool.

    Создаёт детерминированное многоэпоховое расписание по одной фиксированной
    обучающей подвыборке.

    Обучающая подвыборка выбирается из доступного pool один раз. Затем каждая
    эпоха проходит по тем же объектам, при необходимости в новом
    детерминированном порядке. Это соответствует обычному смыслу эпохи:
    повторный проход по тем же выбранным обучающим данным, а не неявный
    пересэмплинг из полного pool.
    """
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if samples_per_epoch <= 0:
        raise ValueError("samples_per_epoch must be positive")
    if n_epochs <= 0:
        raise ValueError("n_epochs must be positive")
    if samples_per_epoch > n_items:
        raise ValueError(
            "samples_per_epoch exceeds the available training pool: "
            f"requested {samples_per_epoch}, available {n_items}"
        )

    rng = np.random.default_rng(seed)

    # EN: Select the training subset exactly once. Reusing the same subset is
    #     what gives the epoch count a clear experimental meaning.
    # RU: Выбираем обучающую подвыборку ровно один раз. Повторное использование
    #     тех же объектов и придаёт числу эпох однозначный экспериментальный смысл.
    pool = np.arange(n_items, dtype=np.int64)
    rng.shuffle(pool)
    subset = pool[:samples_per_epoch].copy()

    epochs: list[np.ndarray] = []
    for _ in range(n_epochs):
        order = subset.copy()
        if shuffle_each_epoch:
            rng.shuffle(order)
        epochs.append(order)

    return np.concatenate(epochs)
