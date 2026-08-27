from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset


GSM_FINAL_RE = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)


def extract_gsm8k_answer(solution: str) -> str:
    match = GSM_FINAL_RE.search(solution)
    if not match:
        return solution.strip()
    return match.group(1).strip()


def _load_split(data_config: dict[str, Any], split: str) -> "Dataset":
    from datasets import load_dataset
    local_path = data_config.get("local_path")
    if local_path:
        suffix = Path(local_path).suffix.lower()
        if suffix not in {".json", ".jsonl", ".csv", ".parquet"}:
            raise ValueError("data.local_path must be JSON, JSONL, CSV, or Parquet")
        loader = "json" if suffix in {".json", ".jsonl"} else suffix.lstrip(".")
        return load_dataset(loader, data_files={split: local_path}, split=split)

    return load_dataset(
        data_config["dataset_name"],
        data_config.get("dataset_config"),
        split=split,
    )


def _prepare_dataset(dataset: "Dataset", data_config: dict[str, Any]) -> "Dataset":
    question_column = data_config.get("question_column", "question")
    answer_column = data_config.get("answer_column", "answer")
    system_prompt = data_config["system_prompt"]
    missing = {question_column, answer_column} - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    def prepare(row: dict[str, Any]) -> dict[str, Any]:
        question = str(row[question_column]).strip()
        raw_solution = str(row[answer_column]).strip()
        answer = extract_gsm8k_answer(raw_solution)
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "question": question,
            "answer": answer,
            "solution": raw_solution,
        }

    return dataset.map(prepare, remove_columns=dataset.column_names)


def _limit(dataset: "Dataset", maximum: int | None) -> "Dataset":
    if maximum is None:
        return dataset
    return dataset.select(range(min(int(maximum), len(dataset))))


def load_train_eval_datasets(
    data_config: dict[str, Any],
) -> tuple["Dataset", "Dataset | None"]:
    train = _load_split(data_config, data_config.get("train_split", "train"))
    eval_split = data_config.get("eval_split")
    validation_fraction = float(data_config.get("validation_fraction", 0.0))
    seed = int(data_config.get("seed", 42))

    if eval_split:
        evaluation = _load_split(data_config, eval_split)
    elif validation_fraction > 0:
        split = train.train_test_split(test_size=validation_fraction, seed=seed)
        train, evaluation = split["train"], split["test"]
    else:
        evaluation = None

    train = _limit(train, data_config.get("max_train_samples"))
    train = _prepare_dataset(train, data_config)
    if evaluation is not None:
        evaluation = _limit(evaluation, data_config.get("max_eval_samples"))
        evaluation = _prepare_dataset(evaluation, data_config)
    return train, evaluation


def load_evaluation_dataset(
    data_config: dict[str, Any], split: str = "test", max_samples: int | None = None
) -> "Dataset":
    dataset = _load_split(data_config, split)
    dataset = _limit(dataset, max_samples)
    return _prepare_dataset(dataset, data_config)
