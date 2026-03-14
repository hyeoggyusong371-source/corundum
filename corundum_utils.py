#!/usr/bin/env python3
# corundum_utils.py
# CORUNDUM shared utilities

import json
import re
import logging
from typing import Any, Dict

log = logging.getLogger("corundum")

try:
    from json_repair import repair_json
    _JSON_REPAIR_OK = True
except ImportError:
    _JSON_REPAIR_OK = False


def safe_parse_json(raw: str, default_fallback: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Parse unstable LLM JSON output safely.

    Order:
      1. Strip <think>...</think> and markdown fences
      2. json-repair (if available)
      3. Depth-tracking parser — matches nested { } correctly
      4. Return default_fallback on full failure
    """
    if default_fallback is None:
        default_fallback = {}

    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    if _JSON_REPAIR_OK:
        try:
            result = repair_json(text, return_objects=True)
            if isinstance(result, dict):
                return result
        except Exception as e:
            log.debug("safe_parse_json: json-repair failed: %s", e)

    start = text.find("{")
    if start == -1:
        log.debug("safe_parse_json: no opening brace found")
        return default_fallback

    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        log.debug("safe_parse_json: brace mismatch")
        return default_fallback

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as e:
        log.debug("safe_parse_json: parse failed: %s", e)
        return default_fallback
