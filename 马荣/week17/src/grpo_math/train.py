from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import GRPOConfig, GRPOTrainer

from grpo_math.config import dump_config, load_config
from grpo_math.data import load_train_eval_datasets
from grpo_math.rewards import REWARD_FUNCTIONS, REWARD_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a math model with GRPO")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="PATH=VALUE",
        help="Override a configuration value; repeat as needed",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def _dtype(precision: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[precision]


def _build_quantization(model_config: dict[str, Any]) -> BitsAndBytesConfig | None:
    if model_config.get("quantization") != "4bit":
        return None
    compute_dtype = _dtype(model_config["precision"])
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _build_lora(lora: dict[str, Any]) -> LoraConfig | None:
    if not lora.get("enabled", False):
        return None
    return LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora.get("dropout", 0.0)),
        bias=lora.get("bias", "none"),
        target_modules=list(lora["target_modules"]),
        task_type="CAUSAL_LM",
    )


def _validate_runtime(config: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GRPO training requires a CUDA GPU; no CUDA device was detected")
    precision = config["model"]["precision"]
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16; set model.precision=float16")

    trainer = config["trainer"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch = (
        world_size
        * int(trainer.get("per_device_train_batch_size", 1))
        * int(trainer.get("gradient_accumulation_steps", 1))
    )
    generations = int(trainer.get("num_generations", 8))
    if effective_batch % generations:
        raise ValueError(
            f"Effective batch size {effective_batch} must be divisible by "
            f"num_generations={generations}"
        )
    if trainer.get("eval_strategy", "no") != "no":
        eval_generations = int(trainer.get("num_generations_eval") or generations)
        effective_eval_batch = world_size * int(trainer.get("per_device_eval_batch_size", 1))
        if effective_eval_batch % eval_generations:
            raise ValueError(
                f"Effective evaluation batch size {effective_eval_batch} must be divisible by "
                f"num_generations_eval={eval_generations}"
            )


def train(config: dict[str, Any], resume_from_checkpoint: str | None = None) -> None:
    _validate_runtime(config)
    model_config = config["model"]
    trainer_values = dict(config["trainer"])
    output_dir = Path(trainer_values.pop("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, output_dir / "resolved_config.yaml")

    seed = int(trainer_values.get("seed", 42))
    set_seed(seed)
    train_dataset, eval_dataset = load_train_eval_datasets(config["data"])
    if trainer_values.get("eval_strategy", "no") != "no" and eval_dataset is None:
        raise ValueError("Evaluation is enabled but no evaluation dataset is configured")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    precision = model_config["precision"]
    trainer_values["bf16"] = precision == "bfloat16"
    trainer_values["fp16"] = precision == "float16"
    trainer_values["model_init_kwargs"] = {
        "dtype": precision,
        "attn_implementation": model_config.get("attn_implementation", "sdpa"),
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "use_cache": not bool(trainer_values.get("gradient_checkpointing", True)),
    }
    reward_weights = [float(config["rewards"][name]) for name in REWARD_NAMES]
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        reward_weights=reward_weights,
        **trainer_values,
    )

    grpo_trainer = GRPOTrainer(
        model=model_config["name_or_path"],
        args=training_args,
        reward_funcs=REWARD_FUNCTIONS,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        quantization_config=_build_quantization(model_config),
        peft_config=_build_lora(config["lora"]),
    )
    result = grpo_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_dir = output_dir / "final_adapter"
    grpo_trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    grpo_trainer.save_state()

    metrics = dict(result.metrics)
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(eval_dataset) if eval_dataset is not None else 0
    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    train(config, args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
