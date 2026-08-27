from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Iterable


STRICT_FORMAT_RE = re.compile(
    r"^\s*<reasoning>\s*(?P<reasoning>.+?)\s*</reasoning>\s*"
    r"<answer>\s*(?P<answer>.+?)\s*</answer>\s*$",
    re.DOTALL,
)
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def completion_text(completion: Any) -> str:
    """Return text from either standard or conversational TRL completions."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(completion or "")


def _extract_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    index = start + len(marker)
    depth = 1
    output: list[str] = []
    while index < len(text) and depth:
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        output.append(char)
        index += 1
    return "".join(output).strip() if depth == 0 else None


def extract_final_answer(text: str) -> str:
    tagged = ANSWER_TAG_RE.findall(text)
    if tagged:
        return tagged[-1].strip()
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    gsm_match = re.search(r"####\s*(.+?)\s*$", text, re.DOTALL)
    if gsm_match:
        return gsm_match.group(1).strip()
    final_match = re.search(
        r"(?:final answer|answer|therefore)\s*(?:is|=|:)\s*([^\n]+)",
        text,
        re.IGNORECASE,
    )
    if final_match:
        return final_match.group(1).strip()
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return nonempty_lines[-1] if nonempty_lines else ""


def _strip_math_wrappers(value: str) -> str:
    value = value.strip().strip("$ ")
    value = value.replace("−", "-").replace("–", "-").replace("，", ",")
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = value.replace(r"\,", "").replace(",", "")
    value = re.sub(r"^(?:USD|RMB)\s*", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".。")
    return value


def _as_number(value: str) -> Fraction | Decimal | None:
    value = _strip_math_wrappers(value)
    percent = value.endswith("%")
    if percent:
        value = value[:-1].strip()

    latex_fraction = re.fullmatch(r"[+-]?\\frac\{([^{}]+)\}\{([^{}]+)\}", value)
    if latex_fraction:
        sign = -1 if value.startswith("-") else 1
        numerator = latex_fraction.group(1).lstrip("+-")
        denominator = latex_fraction.group(2)
        try:
            result: Fraction | Decimal = sign * Fraction(numerator) / Fraction(denominator)
            return result / 100 if percent else result
        except (ValueError, ZeroDivisionError):
            return None

    if re.fullmatch(r"[+-]?\d+\s*/\s*[+-]?\d+", value):
        numerator, denominator = value.split("/", 1)
        try:
            result = Fraction(int(numerator), int(denominator))
            return result / 100 if percent else result
        except (ValueError, ZeroDivisionError):
            return None

    try:
        result = Decimal(value)
        return result / Decimal(100) if percent else result
    except InvalidOperation:
        return None


def answers_equal(prediction: str, reference: str) -> bool:
    prediction = _strip_math_wrappers(extract_final_answer(prediction))
    reference = _strip_math_wrappers(extract_final_answer(reference))
    predicted_number = _as_number(prediction)
    reference_number = _as_number(reference)
    if predicted_number is not None and reference_number is not None:
        return predicted_number == reference_number

    normalize = lambda text: re.sub(r"\s+", "", text).lower()
    return bool(prediction) and normalize(prediction) == normalize(reference)


def correctness_reward(completions: list[Any], answer: list[str], **_: Any) -> list[float]:
    return [
        1.0 if answers_equal(completion_text(completion), reference) else 0.0
        for completion, reference in zip(completions, answer, strict=True)
    ]


def strict_format_reward(completions: list[Any], **_: Any) -> list[float]:
    return [1.0 if STRICT_FORMAT_RE.fullmatch(completion_text(item)) else 0.0 for item in completions]


def reasoning_quality_reward(completions: list[Any], **_: Any) -> list[float]:
    rewards: list[float] = []
    for item in completions:
        match = STRICT_FORMAT_RE.fullmatch(completion_text(item))
        if not match:
            rewards.append(0.0)
            continue
        reasoning = match.group("reasoning").strip()
        has_work = len(reasoning.split()) >= 5 and bool(re.search(r"[=+\-*/]", reasoning))
        rewards.append(1.0 if has_work else 0.0)
    return rewards


def _ngram_repetition_ratio(words: Iterable[str], n: int = 3) -> float:
    tokens = list(words)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def repetition_penalty_reward(completions: list[Any], **_: Any) -> list[float]:
    rewards = []
    for item in completions:
        words = re.findall(r"\w+|[^\w\s]", completion_text(item).lower())
        penalty = -_ngram_repetition_ratio(words)
        rewards.append(0.0 if math.isclose(penalty, 0.0) else penalty)
    return rewards


REWARD_FUNCTIONS = [
    correctness_reward,
    strict_format_reward,
    reasoning_quality_reward,
    repetition_penalty_reward,
]

REWARD_NAMES = [
    "correctness",
    "strict_format",
    "reasoning_quality",
    "repetition_penalty",
]

