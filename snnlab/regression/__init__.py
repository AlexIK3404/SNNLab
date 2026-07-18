"""
Regression utilities for validating framework behavior against frozen references.

Инструменты regression-проверки поведения фреймворка по зафиксированным эталонам.
"""

from .dci_notebook import DCIRegressionReport, run_dci_notebook_regression

__all__ = ["DCIRegressionReport", "run_dci_notebook_regression"]
