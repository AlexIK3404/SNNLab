# Regression checks

Utilities that compare selected framework behavior against frozen historical reference data.

Утилиты, сравнивающие отдельные части поведения фреймворка с зафиксированными историческими эталонными данными.

This package is a validation guard, not the framework specification. Intentional architectural improvements may require introducing a new reference fixture instead of forcing new code to reproduce old behavior.

Этот пакет служит защитой от случайных регрессий, но не является спецификацией фреймворка. Осознанные архитектурные улучшения не обязаны копировать старое поведение; при необходимости создаётся новый эталон.