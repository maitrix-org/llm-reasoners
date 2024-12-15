import argparse


def parse_common_arguments(parser: argparse.ArgumentParser):
    # Task parameters
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name of the task to run. e.g., webarena.<task_id>. Note you have to host the task website somewhere.",
    )
    parser.add_argument("--task_seed", type=int,
                        default=42, help="Seed for the task.")
    # Path parameters
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="results/tree-search",
        help="Directory to save the results.",
    )
    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use in the demo, while you can adapt to any other model from `reasoners.lm`.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for the model."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048, help="Maximum tokens for the model."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "sglang"],
        help="Backend for the model. Currently support `openai` and `sglang`.",
    )
    # Environment parameters
    parser.add_argument(
        "--max_steps",
        type=int,
        default=15,
        help="Maximum steps allowed for the environment.",
    )
    parser.add_argument(
        "--action_set", type=str, default="webarena", help="Action set to use."
    )
    parser.add_argument(
        "--use_axtree",
        type=bool,
        default=True,
        help="Use a11y tree for the observation to LLM.",
    )
    parser.add_argument(
        "--use_html",
        type=bool,
        default=False,
        help="Use HTML for the observation to LLM.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=bool,
        default=False,
        help="Use screenshot for the observation to LLM. Make sure the model can process images if set to `True`.",
    )
    parser.add_argument(
        "--record_video",
        type=bool,
        default=False,
        help="Record the video of the browser.",
    )
