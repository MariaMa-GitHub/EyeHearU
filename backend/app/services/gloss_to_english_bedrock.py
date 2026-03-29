"""
Gloss-to-English rewrite using Amazon Bedrock (Anthropic Claude format).
"""

from __future__ import annotations

import json
import re
from typing import Any

import boto3


def gloss_sequence_to_english_bedrock(
    glosses: list[str],
    *,
    region: str,
    model_id: str,
    timeout_s: float = 20.0,
) -> str:
    """Convert gloss tokens to one natural English sentence using Bedrock."""
    clean = [str(g).strip().replace("_", " ") for g in glosses if str(g).strip()]
    if not clean:
        return ""

    prompt = (
        "You are an ASL gloss to English rewriter. Output exactly one concise natural "
        "English sentence. Preserve meaning. Add needed articles/prepositions/function "
        f"words. Do not add facts.\nGloss sequence: {' '.join(clean)}"
    )

    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=boto3.session.Config(read_timeout=timeout_s, connect_timeout=timeout_s),
    )

    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 80,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }
    resp = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body).encode("utf-8"),
    )
    payload = json.loads(resp["body"].read())
    text = (
        payload.get("content", [{}])[0].get("text", "").strip()
        if isinstance(payload.get("content"), list)
        else ""
    )
    text = _normalize_sentence_case(text)
    if text and text[-1] not in ".?!":
        text += "."
    return text


_KEEP_UPPER = {
    "ASL",
    "AI",
    "AWS",
    "API",
    "USA",
    "UK",
    "EU",
}


def _normalize_sentence_case(text: str) -> str:
    """
    Normalize all-caps model output to sentence case.

    - Lowercase most words
    - Keep common acronyms uppercase
    - Keep standalone pronoun "I" uppercase
    - Capitalize first alphabetic character in the sentence
    """
    s = (text or "").strip()
    if not s:
        return ""

    # If output is mostly uppercase, normalize to lowercase first.
    letters = [c for c in s if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.7:
            s = s.lower()

    # Fix standalone pronoun "i" and common acronyms.
    s = re.sub(r"\bi\b", "I", s)
    for token in _KEEP_UPPER:
        s = re.sub(rf"\b{token.lower()}\b", token, s)

    # Capitalize first alphabetic character only.
    chars = list(s)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = ch.upper()
            break
    return "".join(chars)

