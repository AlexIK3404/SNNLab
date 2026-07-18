from snnlab.i18n import Translator


def test_locale_switch_and_help() -> None:
    translator = Translator("en")
    assert translator.tr("app.language") == "Language"
    assert "Recovery" in translator.help_topic("izhikevich.a")["title"]
    assert "assignment" in translator.help_topic("evaluation.assignment_samples")["title"].lower()
    assert "homeostatic" in translator.help_topic("evaluation.homeostasis_mode")["short"].lower()
    assert "assignment" in translator.help_topic("evaluation.assignment_policy")["title"].lower()

    translator.set_locale("ru")
    assert translator.tr("app.language") == "Язык"
    assert "Временной" in translator.help_topic("izhikevich.a")["title"]
    assert "назначения" in translator.help_topic("evaluation.assignment_samples")["title"].lower()
    assert "гомеостат" in translator.help_topic("evaluation.homeostasis_mode")["short"].lower()
    assert "назначения" in translator.help_topic("evaluation.assignment_policy")["title"].lower()
