from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = {"model", "lora", "data", "rewards", "trainer"}


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    config = copy.deepcopy(loaded)
    for override in overrides or []:
        apply_override(config, override)
    validate_config(config)
    return config


def apply_override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must use dotted.path=value syntax: {expression!r}")
    dotted_key, raw_value = expression.split("=", 1)
    keys = dotted_key.split(".")
    if not all(keys):
        raise ValueError(f"Invalid override key: {dotted_key!r}")

    cursor: dict[str, Any] = config
    for key in keys[:-1]:
        value = cursor.get(key)
        if not isinstance(value, dict):
            raise KeyError(f"Unknown configuration path: {dotted_key!r}")
        cursor = value
    if keys[-1] not in cursor:
        raise KeyError(f"Unknown configuration key: {dotted_key!r}")
    cursor[keys[-1]] = yaml.safe_load(raw_value)


def validate_config(config: dict[str, Any]) -> None:
    missing = REQUIRED_SECTIONS - config.keys()
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    model = config["model"]
    if model.get("precision") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("model.precision must be bfloat16, float16, or float32")
    if model.get("quantization", "none") not in {"4bit", "none", None}:
        raise ValueError("model.quantization must be 4bit or none")
    if model.get("quantization") == "4bit" and not config["lora"].get("enabled", False):
        raise ValueError("4-bit training requires lora.enabled=true")

    trainer = config["trainer"]
    generations = int(trainer.get("num_generations", 8))
    if generations <= 1:
        raise ValueError("trainer.num_generations must be greater than 1 for GRPO")

    weights = config["rewards"]
    expected = {"correctness", "strict_format", "reasoning_quality", "repetition_penalty"}
    if set(weights) != expected:
        raise ValueError(f"rewards must contain exactly these keys: {sorted(expected)}")


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
