from snnlab.evaluation.dci import DCIEvaluationConfig


def test_assignment_policy_invalidates_cached_responses() -> None:
    """Changing assignment source must invalidate response caching.

    Изменение assignment-источника должно инвалидировать кэш откликов.
    """
    full = DCIEvaluationConfig(assignment_policy="full_train_pool")
    disjoint = DCIEvaluationConfig(assignment_policy="exclude_training")
    assert full.response_signature() != disjoint.response_signature()
