from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from grpo_math.config import load_config
from grpo_math.data import load_evaluation_dataset
from grpo_math.rewards import answers_equal, extract_final_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a base model or GRPO LoRA adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None, help="Path to the trained LoRA adapter")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=1, help="Generations per problem")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="outputs/evaluation.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _model_dtype(precision: str) -> torch.dtype:
    if precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if precision in {"bfloat16", "float16"} and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _load_model(config: dict[str, Any], adapter: str | None):
    model_config = config["model"]
    base_model = model_config["name_or_path"]
    if adapter:
        peft_config = PeftConfig.from_pretrained(adapter)
        base_model = peft_config.base_model_name_or_path

    tokenizer_source = adapter or base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=_model_dtype(model_config["precision"]),
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def _collate(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {key: [row[key] for row in rows] for key in rows[0]}


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, float | int]:
    set_seed(args.seed)
    config = load_config(args.config)
    dataset = load_evaluation_dataset(config["data"], args.split, args.max_samples)
    model, tokenizer = _load_model(config, args.adapter)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    top1_correct = 0
    pass_at_k = 0
    records: list[dict[str, Any]] = []
    sampling = args.num_samples > 1

    for batch in loader:
        rendered = [
            tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            for prompt in batch["prompt"]
        ]
        encoded = tokenizer(rendered, return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": sampling,
            "num_return_sequences": args.num_samples,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if sampling:
            generation_kwargs["temperature"] = args.temperature
        generated = model.generate(
            **encoded,
            **generation_kwargs,
        )
        prompt_length = encoded["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)

        for row_index, (question, gold) in enumerate(zip(batch["question"], batch["answer"])):
            start = row_index * args.num_samples
            predictions = decoded[start : start + args.num_samples]
            correctness = [answers_equal(prediction, gold) for prediction in predictions]
            total += 1
            top1_correct += int(correctness[0])
            pass_at_k += int(any(correctness))
            records.append(
                {
                    "question": question,
                    "gold": gold,
                    "predictions": predictions,
                    "extracted_answers": [extract_final_answer(item) for item in predictions],
                    "correct": correctness,
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics: dict[str, float | int] = {
        "num_problems": total,
        "accuracy_at_1": top1_correct / total if total else 0.0,
        f"pass_at_{args.num_samples}": pass_at_k / total if total else 0.0,
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
