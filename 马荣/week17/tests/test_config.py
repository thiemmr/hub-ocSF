import pytest

from grpo_math.config import apply_override, validate_config


def valid_config():
    return {
        "model": {"precision": "bfloat16", "quantization": "4bit"},
        "lora": {"enabled": True},
        "data": {},
        "rewards": {
            "correctness": 1.0,
            "strict_format": 0.1,
            "reasoning_quality": 0.05,
            "repetition_penalty": 0.05,
        },
        "trainer": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_generations": 4,
        },
    }


def test_apply_override_parses_yaml_value():
    config = valid_config()
    apply_override(config, "trainer.num_generations=2")
    assert config["trainer"]["num_generations"] == 2


def test_single_generation_is_rejected():
    config = valid_config()
    config["trainer"]["num_generations"] = 1
    with pytest.raises(ValueError, match="greater than 1"):
        validate_config(config)
