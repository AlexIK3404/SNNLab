# Internationalization (`i18n`)

RU/EN localization and study-mode documentation.

Локализация RU/EN и справка режима изучения.

## Layout

- `resources/` — ordinary interface strings.
- `help/` — extended parameter/concept explanations used by study mode.
- `translator.py` — runtime translation/help lookup.

Stable internal IDs should remain language-independent. Do not use translated UI text as configuration keys or program logic.

Внутренние идентификаторы должны оставаться независимыми от языка. Переведённый текст интерфейса нельзя использовать как ключи конфигурации или элементы логики программы.