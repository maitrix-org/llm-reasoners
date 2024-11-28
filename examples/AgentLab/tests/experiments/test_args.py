from ast import mod
from dataclasses import dataclass
from agentlab.experiments.args import (
    expand_cross_product,
    CrossProd,
    Choice,
    make_progression_study,
    sample_args,
    make_ablation_study,
)


@dataclass
class LLMArgsTest:
    model_name: str = "model1"
    temperature: float = 0.1


@dataclass
class ExpArgsTest:
    llm_args: LLMArgsTest
    task_name: str = "task1"
    n_episode: int = 10


def test_cross_product():
    exp_args = ExpArgsTest(
        n_episode=CrossProd([1, 2, 3]),
        llm_args=LLMArgsTest(
            model_name=CrossProd(["model1", "model2"]),
        ),
    )

    expanded_args_list = expand_cross_product(exp_args)
    assert len(expanded_args_list) == 6

    variables = [(args.n_episode, args.llm_args.model_name) for args in expanded_args_list]
    variables.sort()
    assert variables == [
        (1, "model1"),
        (1, "model2"),
        (2, "model1"),
        (2, "model2"),
        (3, "model1"),
        (3, "model2"),
    ]


def test_cross_product_dict():
    exp_args_dict = dict(
        n_episode=CrossProd([1, 2, 3]),
        llm_args=dict(
            model_name=CrossProd(["model1", "model2"]),
        ),
    )

    expanded_args_list = expand_cross_product(exp_args_dict)
    assert len(expanded_args_list) == 6

    variables = [(args["n_episode"], args["llm_args"]["model_name"]) for args in expanded_args_list]
    variables.sort()
    assert variables == [
        (1, "model1"),
        (1, "model2"),
        (2, "model1"),
        (2, "model2"),
        (3, "model1"),
        (3, "model2"),
    ]


def test_sample():
    exp_args = ExpArgsTest(
        n_episode=Choice([1, 2, 3]),
        llm_args=LLMArgsTest(
            model_name=Choice(["model1", "model2"]),
        ),
    )
    exp_args_sample = sample_args(exp_args, 3)
    assert len(exp_args_sample) == 3
    for exp_args in exp_args_sample:
        assert exp_args.n_episode in [1, 2, 3]
        assert exp_args.llm_args.model_name in ["model1", "model2"]
        assert exp_args.llm_args.temperature == 0.1


def test_sample_and_cross_prod():
    exp_args = ExpArgsTest(
        n_episode=Choice([1, 2, 3]),
        llm_args=LLMArgsTest(
            model_name=CrossProd(["model1", "model2"]),
        ),
    )

    def assert_ok(exp_args_list):
        assert len(exp_args_list) == 6
        for exp_args in exp_args_list:
            assert exp_args.n_episode in [1, 2, 3]
            assert exp_args.llm_args.model_name in ["model1", "model2"]
            assert exp_args.llm_args.temperature == 0.1

    exp_args_list_1 = expand_cross_product(sample_args(exp_args, 3))
    assert_ok(exp_args_list_1)
    exp_args_list_2 = sample_args(expand_cross_product(exp_args), 3)
    assert_ok(exp_args_list_2)


def test_make_progression_study():
    ablation = make_progression_study(
        start_point=LLMArgsTest(
            model_name="model1",
            temperature=0.1,
        ),
        changes=[
            ("model_name", "model2"),
            ("temperature", 0.2),
        ],
    )

    configs = expand_cross_product(
        ExpArgsTest(
            n_episode=CrossProd([1, 2]),
            llm_args=ablation,
        )
    )

    assert len(configs) == 6
    params = [(config.llm_args.model_name, config.llm_args.temperature) for config in configs]
    params = list(set(params))
    params.sort()

    assert params == [("model1", 0.1), ("model2", 0.1), ("model2", 0.2)]


def test_make_ablation_study():
    ablation = make_ablation_study(
        start_point=LLMArgsTest(
            model_name="model1",
            temperature=0.1,
        ),
        changes=[
            ("model_name", "model2"),
            ("temperature", 0.2),
        ],
    )

    configs = expand_cross_product(
        ExpArgsTest(
            n_episode=CrossProd([1, 2]),
            llm_args=ablation,
        )
    )

    assert len(configs) == 6
    params = [(config.llm_args.model_name, config.llm_args.temperature) for config in configs]
    params = list(set(params))
    params.sort()

    assert params == [("model1", 0.1), ("model1", 0.2), ("model2", 0.1)]
