import json

from rag.eval.judge import NO_CONTEXT, Judge


def make_judge(reply):
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return reply if isinstance(reply, str) else json.dumps(reply)

    return Judge(model="fake", generate=generate), prompts


def test_scores_are_parsed_and_clamped():
    judge, _ = make_judge({"answer_correctness": 1.4, "faithfulness": -0.2, "reasoning": "ok"})
    s = judge.score("q", "ref", "ans", ["ctx"])
    assert s == {"answer_correctness": 1.0, "faithfulness": 0.0, "reasoning": "ok"}


def test_faithfulness_forced_null_without_context():
    judge, prompts = make_judge({"answer_correctness": 0.5, "faithfulness": 0.9, "reasoning": ""})
    s = judge.score("q", "ref", "ans", None)
    assert s["faithfulness"] is None
    assert NO_CONTEXT in prompts[0]


def test_prompt_contains_all_parts():
    judge, prompts = make_judge({"answer_correctness": 0, "faithfulness": 0})
    judge.score("THE-Q", "THE-REF", "THE-ANS", ["CTX-1", "CTX-2"])
    for part in ["THE-Q", "THE-REF", "THE-ANS", "CTX-1", "CTX-2"]:
        assert part in prompts[0]


def test_bad_judge_output_is_reported_not_raised():
    judge, _ = make_judge("not json")
    s = judge.score("q", "ref", "ans", ["ctx"])
    assert s["answer_correctness"] is None
    assert "JSONDecodeError" in s["error"]
