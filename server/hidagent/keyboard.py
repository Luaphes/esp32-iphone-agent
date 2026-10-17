"""Translate one text entry into HID-safe keys, leaving Chinese candidates uncommitted."""

import re

from pypinyin import lazy_pinyin


def prepare_text(text: str) -> tuple[str, dict | None]:
    if text and text.isascii() and all(32 <= ord(c) < 127 for c in text):
        return text, None
    if not re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]{1,12}", text):
        raise ValueError("type accepts printable ASCII or 1–12 Chinese characters; split mixed text and punctuation")
    keys = "'".join(lazy_pinyin(text))
    if not re.fullmatch(r"[a-z]+(?:'[a-z]+)*", keys):
        raise ValueError("no keyboard-safe Pinyin for this text; use a shorter phrase or the onscreen keyboard")
    return keys, {"mode": "pinyin", "target": text, "keys": keys, "needs_confirmation": True}
