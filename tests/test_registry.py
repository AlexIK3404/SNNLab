from snnlab.core import architectures, backends, neuron_models, numerical_methods
from snnlab.i18n import Translator


def test_builtin_catalog_has_stable_localizable_ids() -> None:
    assert "dci" in architectures
    assert "reservoir" in architectures
    assert "python_cpu" in backends
    assert "izhikevich" in neuron_models
    assert "semi_euler" in numerical_methods

    descriptor = numerical_methods.descriptor("semi_euler")
    translator = Translator("ru")
    assert translator.tr(descriptor.display_name_key) == "Полуявный Эйлер"
