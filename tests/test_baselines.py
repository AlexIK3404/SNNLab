from snnlab.baselines import available_baselines, load_baseline


def test_dci_mnist_v0_is_packaged_and_self_consistent() -> None:
    assert "dci_mnist_v0" in available_baselines()
    baseline = load_baseline("dci_mnist_v0")
    assert baseline["framework_version"] == "0.3.5"
    assert baseline["training"]["total_presentations"] == (
        baseline["training"]["samples_per_epoch"] * baseline["training"]["epochs"]
    )
    assert baseline["reported_result"]["balanced_topk_accuracy"] == 0.554
