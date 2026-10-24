"""The two languages of the page and the 運作說明 page: Traditional Chinese and English.

Every word a person sees is written in both, side by side where it is used: a string in code is
pick(lang, "中文", "English"), and text kept as data is a {"zh": ..., "en": ...} dict. The page starts
in the browser's language (from_locale) and the switch at its top right changes it.

What the model reads in a run (tool errors, refusals) is written in the language the run started in;
the system prompt is the same in both, so the model behaves the same whichever language the page is in.
"""

LANGS = {"zh": "中文", "en": "English"}  # language -> its name on the switch


def pick(lang: str, zh: str, en: str) -> str:
    return en if lang == "en" else zh


def from_locale(locale: str | None) -> str:
    """The language for a browser's navigator.language: Chinese for any zh-*, English for the rest."""
    return "zh" if (locale or "").lower().startswith("zh") else "en"
