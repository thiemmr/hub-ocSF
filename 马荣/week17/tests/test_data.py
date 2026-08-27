from grpo_math.data import extract_gsm8k_answer


def test_extract_gsm_answer():
    assert extract_gsm8k_answer("work\n#### 1,234") == "1,234"


def test_non_gsm_answer_is_preserved():
    assert extract_gsm8k_answer("42") == "42"

