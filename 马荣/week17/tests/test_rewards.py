from grpo_math.rewards import (
    answers_equal,
    correctness_reward,
    extract_final_answer,
    repetition_penalty_reward,
    strict_format_reward,
)


def conversational(text: str):
    return [{"role": "assistant", "content": text}]


def test_nested_boxed_answer_extraction():
    assert extract_final_answer(r"work \boxed{\frac{1}{3}}") == r"\frac{1}{3}"


def test_numeric_answer_equivalence():
    assert answers_equal("<answer>1/2</answer>", "0.5")
    assert answers_equal("<answer>50%</answer>", "0.5")
    assert answers_equal("<answer>1,200</answer>", "1200")


def test_strict_format_reward():
    good = conversational("<reasoning>2 + 2 = 4 clearly</reasoning><answer>4</answer>")
    bad = conversational("The answer is 4")
    assert strict_format_reward([good, bad]) == [1.0, 0.0]


def test_correctness_reward_accepts_conversational_completion():
    completion = conversational("<reasoning>6 / 2 = 3</reasoning><answer>3</answer>")
    assert correctness_reward([completion], answer=["3"]) == [1.0]


def test_repetition_is_penalized():
    repeated = conversational("one two three one two three one two three")
    clean = conversational("one two three four five six seven")
    rewards = repetition_penalty_reward([repeated, clean])
    assert rewards[0] < rewards[1]

