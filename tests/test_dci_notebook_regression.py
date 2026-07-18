from snnlab.regression import run_dci_notebook_regression


def test_framework_matches_clean_notebook_reference() -> None:
    """
    Verifies Stage-1 DCI against the frozen clean-notebook reference.

    Проверяет Stage-1 DCI по зафиксированному эталону clean-блокнота.
    """
    report = run_dci_notebook_regression()
    failures = [f"{check.name}: {check.details}" for check in report.checks if not check.passed]
    assert report.passed, "\n".join(failures)
