from __future__ import annotations

import math
import re
from collections import Counter


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def word_tokens(s: str) -> list[str]:
    s = normalize_text(s)
    return re.findall(r"[a-z0-9']+", s)


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(ref) else 0.0


def token_precision_recall_f1(pred: str, ref: str) -> tuple[float, float, float]:
    p = Counter(word_tokens(pred))
    r = Counter(word_tokens(ref))
    overlap = sum((p & r).values())
    p_total = sum(p.values())
    r_total = sum(r.values())
    prec = overlap / p_total if p_total else 0.0
    rec = overlap / r_total if r_total else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu_n_precision(pred: str, ref: str, n: int = 4) -> float:
    p_toks = word_tokens(pred)
    r_toks = word_tokens(ref)
    p_ngrams = Counter(_ngrams(p_toks, n))
    r_ngrams = Counter(_ngrams(r_toks, n))
    total = sum(p_ngrams.values())
    if total == 0:
        return 0.0
    overlap = sum((p_ngrams & r_ngrams).values())
    return overlap / total


def sentence_bleu(pred: str, ref: str, max_n: int = 4) -> float:
    p_toks = word_tokens(pred)
    r_toks = word_tokens(ref)
    if not p_toks or not r_toks:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        p = bleu_n_precision(pred, ref, n)
        if p == 0:
            return 0.0
        precisions.append(p)
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    bp = 1.0
    if len(p_toks) < len(r_toks):
        bp = math.exp(1 - (len(r_toks) / max(len(p_toks), 1)))
    return bp * geo_mean


def rouge_l_recall(pred: str, ref: str) -> float:
    p = word_tokens(pred)
    r = word_tokens(ref)
    if not r:
        return 0.0
    lcs = _lcs_len(p, r)
    return lcs / len(r)


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def capitalization_ok(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    first_alpha = next((c for c in s if c.isalpha()), "")
    return 1.0 if first_alpha and first_alpha.isupper() else 0.0


def punctuation_ok(s: str) -> float:
    s = (s or "").strip()
    return 1.0 if s.endswith((".", "!", "?")) else 0.0

