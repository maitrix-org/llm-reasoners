"""
Code utils for data preprocessing
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

N_TESTSET_PER_DATASET = 512  # per dataset
MAX_N_TRAINSET_PER_DATASET = 50000
EMPTY_RETURN = {
    "data_source": None,
    "prompt": None,
    "ability": None,
    "reward_model": None,
    "extra_info": None
}
SYSTEM_PROMPT = """You are a helpful programming assistant. \
The user will ask you a question and you as the assistant solve it. \
The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. \
The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively."""
PY_IMPORTS = "import heapq\nfrom math import floor, gcd\nimport random\nimport sys\nfrom typing import *\nfrom functools import *\nimport collections\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom string import *\nimport math\nimport datetime\ninf = float('inf')\n"
BLOCK_LIBS = [
    # "requests",
    # "subprocess",
    # "multiprocessing",
    # "threading",
    # "os",
    # "sys",
    "fake-useragent",
    "keras",
    "socket",
    "torch",
    "scipy",
    "sklearn",
    "cv2",
    "scipy",
    "imageio",
    "sphinx-pyproject",
    "xgboost",
    "tweepy",
    "flask",
    "matplotlib",
    "pillow",
    "seaborn",
    "smtpd",
    "docker",
    "psutil",
    "paramiko",
    "pickle",
    "shutil",
    "telnetlib",
    "ftplib",
    "pyautogui",
    "pexpect",
    "asyncssh",
    "cryptography",
    "pycrypto",
    "scapy",
    "nmap",
    "pwntools",
    "webbrowser",
    "ctypes",
    "tempfile",
]
BLOCK_USAGES = [
    "open(",
    "exec(",
    "eval(",
    "system(",
    "popen(",
    "importlib",
    "globals(",
    "locals(",
    "breakpoint(",
    "__builtins__",
    "compile(",
    "fork(",
    "chmod(",
]
MAX_PROMPT_LENGTH = 2048


def minimize_stdio(inputs, outputs, max_n_tests=8):
    stdin_list = []
    stdout_list = []
    for stdin, stdout in zip(inputs, outputs):
        if isinstance(stdin, list):
            stdin = "\n".join(stdin)
        if isinstance(stdout, list):
            stdout = "\n".join(stdout)
        if sys.getsizeof(stdin) > 4 * 1024:
            continue
        stdout.replace("\r\n", "\n")
        stdin_list.append(stdin)
        stdout_list.append(stdout)

    zipped = sorted(zip(stdin_list, stdout_list), key=lambda x: sys.getsizeof(x[0]))

    if not zipped:
        print("No tests found!")
        return [], []

    sorted_stdin, sorted_stdout = zip(*zipped)
    return list(sorted_stdin[:max_n_tests]), list(sorted_stdout[:max_n_tests])


# Plot the filter counts
def plot_hist(
    data_dict,
    file_path,
    title,
    xlabel,
    ylabel,
    figsize=(10, 6),
    plot_type="bar",
    color=None,
    sort_by=None,
    max_categories=None,
    rotation=45,
    dpi=300,
    log_scale=False,
    grid=True,
    font_size=10,
):
    """
    Plot the histogram of filter counts from a dictionary.

    Args:
        data_dict (dict): Dictionary mapping categories to counts
        file_path (str): Path to save the output figure
        title (str): Title of the plot
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        figsize (tuple): Figure size in inches (width, height)
        plot_type (str): Type of plot ('bar', 'barh', 'pie')
        color (str or list): Color or color map for the plot
        sort_by (str): How to sort categories ('key', 'value', 'value_desc', None)
        max_categories (int): Maximum number of categories to display (None for all)
        rotation (int): Rotation angle for x-tick labels
        dpi (int): DPI for the output figure
        log_scale (bool): Use log scale for y-axis
        grid (bool): Show grid lines
        font_size (int): Base font size for the plot

    Returns:
        None: The figure is saved to file_path
    """
    try:
        plt.figure(figsize=figsize)

        # Handle empty dictionary
        if not data_dict:
            plt.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                fontsize=font_size + 4,
            )
            plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            return

        # Process data
        items = list(data_dict.items())

        # Sort data if requested
        if sort_by == "key":
            items.sort(key=lambda x: x[0])
        elif sort_by == "value":
            items.sort(key=lambda x: x[1])
        elif sort_by == "value_desc":
            items.sort(key=lambda x: x[1], reverse=True)

        # Limit number of categories if requested
        if max_categories and len(items) > max_categories:
            items = items[: max_categories - 1]
            # Add "Other" category if needed
            if max_categories < len(data_dict):
                other_sum = sum(v for k, v in data_dict.items() if k not in dict(items))
                items.append(("Other", other_sum))

        labels, counts = zip(*items) if items else ([], [])

        # Create the appropriate plot
        if plot_type == "bar":
            plt.bar(labels, counts, color=color)
            if log_scale and any(counts):
                plt.yscale("log")
            plt.xticks(rotation=rotation, ha="right", fontsize=font_size)
            plt.yticks(fontsize=font_size)
        elif plot_type == "barh":
            plt.barh(labels, counts, color=color)
            if log_scale and any(counts):
                plt.xscale("log")
            plt.yticks(fontsize=font_size)
            plt.xticks(fontsize=font_size)
        elif plot_type == "pie":
            plt.pie(
                counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=color
            )
            plt.axis("equal")

        # Set labels and title
        plt.xlabel(xlabel, fontsize=font_size + 2)
        plt.ylabel(ylabel, fontsize=font_size + 2)
        plt.title(title, fontsize=font_size + 4)

        if grid and plot_type != "pie":
            plt.grid(alpha=0.3)

        plt.tight_layout()

        # Ensure directory exists
        import os

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Save figure
        plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
        plt.close()

    except Exception as e:
        import logging

        logging.error(f"Error plotting histogram: {str(e)}")
        # Create a minimal error plot
        plt.figure(figsize=(8, 6))
        plt.text(
            0.5,
            0.5,
            f"Error creating plot: {str(e)}",
            ha="center",
            va="center",
            color="red",
        )
        plt.tight_layout()
        plt.savefig(file_path, dpi=dpi)
        plt.close()
