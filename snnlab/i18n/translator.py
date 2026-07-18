from __future__ import annotations

from importlib.resources import files
from typing import Any

import yaml


class Translator:
    """
    Loads UI strings and contextual help for the selected locale.

    The class is intentionally GUI-agnostic. A future RU/EN switch can call
    set_locale(...) and then refresh visible widgets from the same catalogs.

    Загружает строки интерфейса и контекстную справку для выбранной локали.

    Класс намеренно не зависит от GUI. Будущий переключатель RU/EN сможет
    вызвать set_locale(...) и обновить виджеты из тех же каталогов.
    """

    SUPPORTED_LOCALES = ("en", "ru")

    def __init__(self, locale: str = "en"):
        self._locale = "en"
        self._strings: dict[str, Any] = {}
        self._fallback_strings = self._load_yaml("resources", "en.yaml")
        self._help: dict[str, Any] = {}
        self._fallback_help = self._load_yaml("help", "en.yaml")
        self.set_locale(locale)

    @property
    def locale(self) -> str:
        return self._locale

    @classmethod
    def available_locales(cls) -> tuple[str, ...]:
        return cls.SUPPORTED_LOCALES

    def set_locale(self, locale: str) -> None:
        """
        Switches the active locale at runtime.

        Переключает активную локаль во время работы приложения.
        """
        if locale not in self.SUPPORTED_LOCALES:
            raise ValueError(f"Unsupported locale {locale!r}. Available: {self.SUPPORTED_LOCALES}")
        self._locale = locale
        self._strings = self._load_yaml("resources", f"{locale}.yaml")
        self._help = self._load_yaml("help", f"{locale}.yaml")

    def tr(self, key: str, **kwargs: Any) -> str:
        """
        Resolves a dotted translation key and formats placeholders.

        Разрешает dotted-ключ перевода и форматирует placeholders.
        """
        value = self._resolve(self._strings, key)
        if value is None:
            value = self._resolve(self._fallback_strings, key)
        if value is None:
            return key
        return str(value).format(**kwargs)

    def help_topic(self, topic_id: str) -> dict[str, Any]:
        """
        Returns localized structured help for a UI parameter or term.

        Возвращает локализованную структурированную справку для параметра или термина UI.
        """
        topic = self._resolve(self._help, topic_id)
        if topic is None:
            topic = self._resolve(self._fallback_help, topic_id)
        if topic is None:
            raise KeyError(f"Unknown help topic {topic_id!r}")
        if not isinstance(topic, dict):
            raise TypeError(f"Help topic {topic_id!r} must be a mapping")
        return dict(topic)

    @staticmethod
    def _resolve(mapping: dict[str, Any], dotted_key: str) -> Any | None:
        current: Any = mapping
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    @staticmethod
    def _load_yaml(folder: str, filename: str) -> dict[str, Any]:
        resource = files("snnlab.i18n").joinpath(folder, filename)
        with resource.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Localization file {filename} must contain a mapping")
        return data
