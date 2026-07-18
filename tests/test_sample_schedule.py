import numpy as np
import pytest

from snnlab.experiments.base import create_epoch_schedule, extend_sample_schedule


def test_extend_without_repeats_excludes_existing_indices() -> None:
    existing = np.array([0, 2, 4], dtype=np.int64)
    combined = extend_sample_schedule(
        existing,
        n_items=8,
        additional_samples=3,
        seed=52,
        allow_repeats=False,
    )
    assert combined.shape == (6,)
    assert len(np.unique(combined)) == 6
    assert set(combined[:3]) == {0, 2, 4}


def test_extend_without_repeats_fails_when_dataset_is_exhausted() -> None:
    with pytest.raises(ValueError):
        extend_sample_schedule(
            np.array([0, 1, 2], dtype=np.int64),
            n_items=4,
            additional_samples=2,
            seed=1,
            allow_repeats=False,
        )


def test_epoch_schedule_reuses_same_subset() -> None:
    """Each epoch must revisit the same fixed subset in a deterministic order.

    Каждая эпоха должна повторно проходить по одной фиксированной подвыборке
    в детерминированном порядке.
    """
    schedule = create_epoch_schedule(
        n_items=20,
        samples_per_epoch=7,
        n_epochs=3,
        seed=52,
    )
    assert schedule.shape == (21,)
    epochs = schedule.reshape(3, 7)
    assert set(epochs[0]) == set(epochs[1]) == set(epochs[2])
    assert np.array_equal(
        schedule,
        create_epoch_schedule(
            n_items=20,
            samples_per_epoch=7,
            n_epochs=3,
            seed=52,
        ),
    )
