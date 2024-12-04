import bgym
import pytest
import ray
from agentlab.experiments.graph_execution_ray import execute_task_graph
from agentlab.experiments.exp_utils import MockedExpArgs, add_dependencies

TASK_TIME = 3


def test_execute_task_graph():
    # Define a list of ExpArgs with dependencies
    exp_args_list = [
        MockedExpArgs(exp_id="task1", depends_on=[]),
        MockedExpArgs(exp_id="task2", depends_on=["task1"]),
        MockedExpArgs(exp_id="task3", depends_on=["task1"]),
        MockedExpArgs(exp_id="task4", depends_on=["task2", "task3"]),
    ]

    ray.init(num_cpus=4)
    results = execute_task_graph(exp_args_list)
    ray.shutdown()

    exp_args_list = [results[task_id] for task_id in ["task1", "task2", "task3", "task4"]]

    # Verify that all tasks were executed in the proper order
    assert exp_args_list[0].start_time < exp_args_list[1].start_time
    assert exp_args_list[0].start_time < exp_args_list[2].start_time
    assert exp_args_list[1].end_time < exp_args_list[3].start_time
    assert exp_args_list[2].end_time < exp_args_list[3].start_time

    # Verify that parallel tasks (task2 and task3) started within a short time of each other
    parallel_start_diff = abs(exp_args_list[1].start_time - exp_args_list[2].start_time)
    print(f"parallel_start_diff: {parallel_start_diff}")
    assert parallel_start_diff < 1.5  # Allow for a small delay

    # Ensure that the entire task graph took the expected amount of time
    total_time = exp_args_list[-1].end_time - exp_args_list[0].start_time
    assert (
        total_time >= TASK_TIME * 3
    )  # Since the critical path involves at least 1.5 seconds of work


def test_add_dependencies():
    # Prepare a simple list of ExpArgs

    def make_exp_args(task_name, exp_id):
        return bgym.ExpArgs(
            agent_args=None, env_args=bgym.EnvArgs(task_name=task_name), exp_id=exp_id
        )

    exp_args_list = [
        make_exp_args("task1", "1"),
        make_exp_args("task2", "2"),
        make_exp_args("task3", "3"),
    ]

    # Define simple task_dependencies
    task_dependencies = {"task1": ["task2"], "task2": [], "task3": ["task1"]}

    # Call the function
    modified_list = add_dependencies(exp_args_list, task_dependencies)

    # Verify dependencies
    assert modified_list[0].depends_on == ("2",)  # task1 depends on task2
    assert modified_list[1].depends_on == ()  # task2 has no dependencies
    assert modified_list[2].depends_on == ("1",)  # task3 depends on task1

    # assert raise if task_dependencies is wrong
    task_dependencies = {"task1": ["task2"], "task2": [], "task4": ["task3"]}
    with pytest.raises(ValueError):
        add_dependencies(exp_args_list, task_dependencies)


if __name__ == "__main__":
    test_execute_task_graph()
    # test_add_dependencies()
