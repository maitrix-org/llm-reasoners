import base64
import os
import traceback
from copy import deepcopy
from io import BytesIO
from logging import warning
from pathlib import Path

import gradio as gr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attr import dataclass
from browsergym.experiments.loop import ExpResult, StepInfo
from langchain.schema import BaseMessage, HumanMessage
from openai import OpenAI
from PIL import Image

from agentlab.analyze import inspect_results
from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.experiments.study import get_most_recent_study
from agentlab.llm.chat_api import make_system_message, make_user_message
from agentlab.llm.llm_utils import BaseMessage as AgentLabBaseMessage
from agentlab.llm.llm_utils import Discussion

select_dir_instructions = "Select Experiment Directory"
AGENT_NAME_KEY = "agent.agent_name"
TASK_NAME_KEY = "env.task_name"
TASK_SEED_KEY = "env.task_seed"


def display_table(df: pd.DataFrame):
    df = df.copy()
    df.columns = clean_column_names(df.columns)
    df.index.names = clean_column_names(df.index.names)
    return df


def remove_args_from_col(df: pd.DataFrame):
    df.columns = [col.replace("_args", "") for col in df.columns]
    df.index.names = [col.replace("_args", "") for col in df.index.names]
    return df


def clean_column_names(col_list):
    # col_list = [col.replace("_args", "") for col in col_list]
    col_list = [col.replace(".", ".\n") for col in col_list]  # adding space for word wrap
    # col_list = [col.replace("_", " ") for col in col_list]
    return col_list


class ClickMapper:
    def __init__(self, ax: plt.Axes, step_times: list[float]):
        self.ax = ax
        self.step_times = step_times

    def to_time(self, x_pix_coord):
        x_time_coord, _ = self.ax.transData.inverted().transform((x_pix_coord, 0))
        return x_time_coord

    def to_step(self, x_pix_coord):
        x_time_coord = self.to_time(x_pix_coord)
        return np.searchsorted(self.step_times, x_time_coord)


@dataclass
class EpisodeId:
    agent_id: str = None
    task_name: str = None
    seed: int = None


@dataclass
class StepId:
    episode_id: EpisodeId = None
    step: int = None


@dataclass
class Info:
    results_dir: Path = None  # to root directory of all experiments
    exp_list_dir: Path = None  # the path of the currently selected experiment
    result_df: pd.DataFrame = None  # the raw loaded df
    agent_df: pd.DataFrame = None  # the df filtered for selected agent
    tasks_df: pd.DataFrame = None  # the unique tasks for selected agent
    exp_result: ExpResult = None  # the selected episode
    click_mapper: ClickMapper = None  # mapping from profiler click to step
    step: int = None  # currently selected step
    active_tab: str = "Screenshot"  # currently selected observation tab
    agent_id_keys: list[str] = None  # the list of columns identifying an agent

    def update_exp_result(self, episode_id: EpisodeId):
        if self.result_df is None or episode_id.task_name is None or episode_id.seed is None:
            self.exp_result = None

        # find unique row for task_name and seed
        result_df = self.agent_df.reset_index(inplace=False)
        sub_df = result_df[
            (result_df[TASK_NAME_KEY] == episode_id.task_name)
            & (result_df[TASK_SEED_KEY] == episode_id.seed)
        ]
        if len(sub_df) == 0:
            self.exp_result = None
            raise ValueError(
                f"Could not find task_name: {episode_id.task_name} and seed: {episode_id.seed}"
            )

        if len(sub_df) > 1:
            warning(
                f"Found multiple rows for task_name: {episode_id.task_name} and seed: {episode_id.seed}. Using the first one."
            )

        exp_dir = sub_df.iloc[0]["exp_dir"]
        print(exp_dir)
        self.exp_result = ExpResult(exp_dir)
        self.step = 0

    def get_agent_id(self, row: pd.Series):
        agent_id = []
        for key in self.agent_id_keys:
            agent_id.append((key, row[key]))
        return agent_id

    def filter_agent_id(self, agent_id: list[tuple]):
        # query_str = " & ".join([f"`{col}` == {repr(val)}" for col, val in agent_id])
        # agent_df = info.result_df.query(query_str)

        agent_df = self.result_df.reset_index(inplace=False)
        agent_df.set_index(TASK_NAME_KEY, inplace=True)

        for col, val in agent_id:
            col = col.replace(".\n", ".")
            agent_df = agent_df[agent_df[col] == val]
        self.agent_df = agent_df


info = Info()


css = """
.my-markdown {
    max-height: 400px;
    overflow-y: auto;
}
.error-report {
    max-height: 700px;
    overflow-y: auto;
}
.my-code-view {
    max-height: 300px;
    overflow-y: auto;
}
code {
    white-space: pre-wrap;
}
th {
    white-space: normal !important;
    word-wrap: break-word !important;
}
"""


def run_gradio(results_dir: Path):
    """
    Run Gradio on the selected experiments saved at savedir_base.

    """
    global info
    info.results_dir = results_dir

    with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
        agent_id = gr.State(value=None)
        episode_id = gr.State(value=EpisodeId())
        agent_task_id = gr.State(value=None)
        step_id = gr.State(value=None)

        with gr.Accordion("Help", open=False):
            gr.Markdown(
                """\
# Agent X-Ray

1. **Select your experiment directory**. You may refresh the list of directories by
clicking the refresh button.

2. **Select your episode**: Chose a triplet (agent, task, seed).

    1. **Select Agent**: Click on a row of the table to select your agent

    2. **Select Task**: Select the task you want to analyze, this will trigger
       an update of the available seeds.

    3. **Select the Seed**: You might have multiple repetition for a given task,
       you will be able to select the seed you want to analyze.

3. **Select the step**: Once your episode is selected, you can select the step
   by clicking on the profiling image. This will trigger the update of the the
   information on the corresponding step.

4. **Select a tab**: You can select different visualization by clicking on the tabs.
"""
            )
        with gr.Row():

            exp_dir_choice = gr.Dropdown(
                choices=get_directory_contents(results_dir),
                value=select_dir_instructions,
                label="Experiment Directory",
                show_label=False,
                scale=6,
                container=False,
            )
            refresh_button = gr.Button("↺", scale=0, size="sm")

        with gr.Tabs():
            with gr.Tab("Select Agent"):
                with gr.Accordion("Agent Selector (click for help)", open=False):
                    gr.Markdown(
                        """\
    Click on a row to select an agent. It will trigger the update of other
    fields.
    
    The update mechanism is somewhat flacky, please help figure out why (or is it just gradio?).
    """
                    )
                agent_table = gr.DataFrame(max_height=500, show_label=False, interactive=False)
            with gr.Tab("Select Task and Seed", id="Select Task"):
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Row():  # combining the title (help) and the refresh button
                            with gr.Accordion("Task Selector (click for help)", open=False):
                                gr.Markdown(
                                    """\
        Click on a row to select a task. It will trigger the update of other fields.

        The update mechanism is somewhat flacky, please help figure out why (or is it just gradio?).
        """
                                )
                            refresh_results_button = gr.Button("↺", scale=0, size="sm")

                        task_table = gr.DataFrame(
                            max_height=500,
                            show_label=False,
                            interactive=False,
                            elem_id="task_table",
                        )

                    with gr.Column(scale=2):
                        with gr.Accordion("Seed Selector (click for help)", open=False):
                            gr.Markdown(
                                """\
    Click on a row to select a seed. It will trigger the update of other fields.

    The update mechanism is somewhat flacky, please help figure out why (or is it just gradio?).
    """
                            )

                        seed_table = gr.DataFrame(
                            max_height=500,
                            show_label=False,
                            interactive=False,
                            elem_id="seed_table",
                        )

            with gr.Tab("Constants and Variables"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Accordion("Constants", open=False):
                            gr.Markdown(
                                """\
    Constants are the parameters that are the same for **all** episodes of
    **all** agents. They are displayed as a table with the name and value of the
    constant."""
                            )
                        constants = gr.DataFrame(
                            max_height=500, show_label=False, interactive=False
                        )
                    with gr.Column(scale=2):
                        with gr.Accordion("Variables", open=False):
                            gr.Markdown(
                                """\
    Variables are the parameters that can change between episodes of an agent.
    They are displayed as a table with the name, value and count of unique
    values. A maximum of 3 different values are displayed."""
                            )
                        variables = gr.DataFrame(
                            max_height=500, show_label=False, interactive=False
                        )
            with gr.Tab("Global Stats"):
                global_stats = gr.DataFrame(max_height=500, show_label=False, interactive=False)

            with gr.Tab("Error Report"):
                error_report = gr.Markdown(elem_classes="error-report", show_copy_button=True)
        with gr.Row():
            episode_info = gr.Markdown(label="Episode Info", elem_classes="my-markdown")
            action_info = gr.Markdown(label="Action Info", elem_classes="my-markdown")
            state_error = gr.Markdown(label="Next Step Error", elem_classes="my-markdown")

        profiling_gr = gr.Image(
            label="Profiling", show_label=False, interactive=False, show_download_button=False
        )

        gr.HTML(
            """
<style>
    .code-container {
        height: 700px;  /* Set the desired height */
        overflow: auto;  /* Enable scrolling */
    }
</style>
"""
        )
        with gr.Tabs() as tabs:
            code_args = dict(interactive=False, elem_classes=["code-container"], show_label=False)
            with gr.Tab("Screenshot") as tab_screenshot:
                som_or_not = gr.Dropdown(
                    choices=["Raw Screenshots", "SOM Screenshots"],
                    label="Screenshot Type",
                    value="Raw Screenshots",
                    show_label=False,
                    container=False,
                    interactive=True,
                    scale=0,
                )
                screenshot = gr.Image(
                    show_label=False, interactive=False, show_download_button=False
                )

            with gr.Tab("Screenshot Pair") as tab_screenshot_pair:
                with gr.Row():
                    screenshot1 = gr.Image(
                        show_label=False, interactive=False, show_download_button=False
                    )
                    screenshot2 = gr.Image(
                        show_label=False, interactive=False, show_download_button=False
                    )
            with gr.Tab("Screenshot Gallery") as tab_screenshot_gallery:
                screenshot_gallery = gr.Gallery(
                    columns=2,
                    show_download_button=False,
                    show_label=False,
                    object_fit="contain",
                    preview=True,
                )

            with gr.Tab("DOM HTML") as tab_html:
                html_code = gr.Code(language="html", **code_args)

            with gr.Tab("Pruned DOM HTML") as tab_pruned_html:
                pruned_html_code = gr.Code(language="html", **code_args)

            with gr.Tab("AXTree") as tab_axtree:
                axtree_code = gr.Code(language=None, **code_args)

            with gr.Tab("Chat Messages") as tab_chat:
                chat_messages = gr.Markdown()

            with gr.Tab("Task Error") as tab_error:
                task_error = gr.Markdown()

            with gr.Tab("Logs") as tab_logs:
                logs = gr.Code(language=None, **code_args)

            with gr.Tab("Stats") as tab_stats:
                stats = gr.DataFrame(max_height=500, show_label=False, interactive=False)

            with gr.Tab("Agent Info HTML") as tab_agent_info_html:
                with gr.Row():
                    screenshot1_agent = gr.Image(
                        show_label=False, interactive=False, show_download_button=False
                    )
                    screenshot2_agent = gr.Image(
                        show_label=False, interactive=False, show_download_button=False
                    )
                agent_info_html = gr.HTML()

            with gr.Tab("Agent Info MD") as tab_agent_info_md:
                agent_info_md = gr.Markdown()

            with gr.Tab("Prompt tests") as tab_prompt_tests:
                with gr.Row():
                    prompt_markdown = gr.Textbox(
                        value="",
                        label="",
                        show_label=False,
                        interactive=False,
                        elem_id="prompt_markdown",
                    )
                    with gr.Column():
                        prompt_tests_textbox = gr.Textbox(
                            value="",
                            label="",
                            show_label=False,
                            interactive=True,
                            elem_id="prompt_tests_textbox",
                        )
                        submit_button = gr.Button(value="Submit")
                    result_box = gr.Textbox(
                        value="", label="Result", show_label=True, interactive=False
                    )

                # Define the interaction
                submit_button.click(
                    fn=submit_action, inputs=prompt_tests_textbox, outputs=result_box
                )

        # Handle Events #
        # ===============#

        refresh_button.click(
            fn=refresh_exp_dir_choices, inputs=exp_dir_choice, outputs=exp_dir_choice
        )

        refresh_results_button.click(
            fn=refresh_exp_dir_choices, inputs=exp_dir_choice, outputs=exp_dir_choice
        )

        exp_dir_choice.change(
            fn=new_exp_dir,
            inputs=exp_dir_choice,
            outputs=[agent_table, agent_id, constants, variables, global_stats, error_report],
        )

        agent_table.select(fn=on_select_agent, inputs=agent_table, outputs=[agent_id])
        task_table.select(fn=on_select_task, inputs=[task_table, agent_id], outputs=agent_task_id)

        agent_id.change(fn=new_agent_id, inputs=agent_id, outputs=[task_table, agent_task_id])
        agent_task_id.change(
            fn=update_seeds, inputs=agent_task_id, outputs=[seed_table, episode_id]
        )
        # seed_gr.change(fn=on_select_seed, inputs=[seed_gr, task_name], outputs=[episode_id])
        seed_table.select(on_select_seed, inputs=[seed_table, agent_task_id], outputs=episode_id)
        step_id.change(fn=update_step_info, outputs=[episode_info, action_info, state_error])
        episode_id.change(fn=new_episode, inputs=[episode_id], outputs=[profiling_gr, step_id])
        profiling_gr.select(select_step, inputs=[episode_id], outputs=step_id)

        # Update all tabs on step change, but only actually update the active
        # tab. This helps keeping the UI responsive when selecting a new step.
        step_id.change(
            fn=if_active("Screenshot")(update_screenshot),
            inputs=som_or_not,
            outputs=screenshot,
        )
        step_id.change(
            fn=if_active("Screenshot Pair", 2)(update_screenshot_pair),
            inputs=som_or_not,
            outputs=[screenshot1, screenshot2],
        )
        step_id.change(
            fn=if_active("Screenshot Gallery")(update_screenshot_gallery),
            inputs=som_or_not,
            outputs=[screenshot_gallery],
        )
        screenshot_gallery.select(fn=gallery_step_change, inputs=episode_id, outputs=step_id)
        step_id.change(fn=if_active("DOM HTML")(update_html), outputs=html_code)
        step_id.change(
            fn=if_active("Pruned DOM HTML")(update_pruned_html), outputs=pruned_html_code
        )
        step_id.change(fn=if_active("AXTree")(update_axtree), outputs=axtree_code)
        step_id.change(fn=if_active("Chat Messages")(update_chat_messages), outputs=chat_messages)
        step_id.change(fn=if_active("Task Error")(update_task_error), outputs=task_error)
        step_id.change(fn=if_active("Logs")(update_logs), outputs=logs)
        step_id.change(fn=if_active("Stats")(update_stats), outputs=stats)
        step_id.change(
            fn=if_active("Agent Info HTML", 3)(update_agent_info_html),
            outputs=[agent_info_html, screenshot1_agent, screenshot2_agent],
        )
        step_id.change(fn=if_active("Agent Info MD")(update_agent_info_md), outputs=agent_info_md)
        step_id.change(
            fn=if_active("Prompt tests", 2)(update_prompt_tests),
            outputs=[prompt_markdown, prompt_tests_textbox],
        )

        # In order to handel tabs that were not visible when step was changed,
        # we need to update them individually when the tab is selected
        tab_screenshot.select(fn=update_screenshot, inputs=som_or_not, outputs=screenshot)
        tab_screenshot_pair.select(
            fn=update_screenshot_pair, inputs=som_or_not, outputs=[screenshot1, screenshot2]
        )
        tab_screenshot_gallery.select(
            fn=update_screenshot_gallery, inputs=som_or_not, outputs=[screenshot_gallery]
        )
        tab_html.select(fn=update_html, outputs=html_code)
        tab_pruned_html.select(fn=update_pruned_html, outputs=pruned_html_code)
        tab_axtree.select(fn=update_axtree, outputs=axtree_code)
        tab_chat.select(fn=update_chat_messages, outputs=chat_messages)
        tab_error.select(fn=update_task_error, outputs=task_error)
        tab_logs.select(fn=update_logs, outputs=logs)
        tab_stats.select(fn=update_stats, outputs=stats)
        tab_agent_info_html.select(fn=update_agent_info_html, outputs=agent_info_html)
        tab_agent_info_md.select(fn=update_agent_info_md, outputs=agent_info_md)
        tab_prompt_tests.select(
            fn=update_prompt_tests, outputs=[prompt_markdown, prompt_tests_textbox]
        )

        som_or_not.change(fn=update_screenshot, inputs=som_or_not, outputs=screenshot)

        # keep track of active tab
        tabs.select(tab_select)

    demo.queue()

    do_share = os.getenv("AGENTXRAY_SHARE_GRADIO", "false").lower() == "true"
    port = os.getenv("AGENTXRAY_APP_PORT", None)
    if isinstance(port, str):
        port = int(port)
    demo.launch(server_port=port, share=do_share)


def tab_select(evt: gr.SelectData):
    global info
    info.active_tab = evt.value


def if_active(tab_name, n_out=1):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            global info
            if info.active_tab == tab_name:
                # print("updating: ", fn.__name__)
                return fn(*args, **kwargs)
            else:
                # print("skipping: ", fn.__name__)
                if n_out == 1:
                    return gr.update()
                elif n_out > 1:
                    return (gr.update(),) * n_out

        return wrapper

    return decorator


def update_screenshot(som_or_not: str):
    global info
    return get_screenshot(info, som_or_not=som_or_not)


def get_screenshot(info: Info, step: int = None, som_or_not: str = "Raw Screenshots"):
    if step is None:
        step = info.step
    try:
        is_som = som_or_not == "SOM Screenshots"
        return info.exp_result.get_screenshot(step, som=is_som)
    except FileNotFoundError:
        return None


def update_screenshot_pair(som_or_not: str):
    global info
    s1 = get_screenshot(info, info.step, som_or_not)
    s2 = get_screenshot(info, info.step + 1, som_or_not)
    return s1, s2


def update_screenshot_gallery(som_or_not: str):
    global info
    screenshots = info.exp_result.get_screenshots(som=som_or_not == "SOM Screenshots")
    screenshots_and_label = [(s, f"Step {i}") for i, s in enumerate(screenshots)]
    gallery = gr.Gallery(
        value=screenshots_and_label,
        columns=2,
        show_download_button=False,
        show_label=False,
        object_fit="contain",
        preview=True,
        selected_index=info.step,
    )
    return gallery


def gallery_step_change(evt: gr.SelectData, episode_id: EpisodeId):
    global info
    info.step = evt.index
    return StepId(episode_id=episode_id, step=evt.index)


def update_html():
    return get_obs(key="dom_txt", default="No DOM HTML")


def update_pruned_html():
    return get_obs(key="pruned_html", default="No Pruned HTML")


def update_axtree():
    return get_obs(key="axtree_txt", default="No AXTree")


def update_chat_messages():
    global info
    agent_info = info.exp_result.steps_info[info.step].agent_info
    chat_messages = agent_info.get("chat_messages", ["No Chat Messages"])
    if isinstance(chat_messages, Discussion):
        return chat_messages.to_markdown()
    messages = []  # TODO(ThibaultLSDC) remove this at some point
    for i, m in enumerate(chat_messages):
        if isinstance(m, BaseMessage):  # TODO remove once langchain is deprecated
            m = m.content
        elif isinstance(m, dict):
            m = m.get("content", "No Content")
        messages.append(f"""# Message {i}\n```\n{m}\n```\n\n""")
    return "\n".join(messages)


def update_task_error():
    global info
    try:
        stack_trace = info.exp_result.summary_info.get("stack_trace", None)
        return f"""{code(stack_trace)}"""
    except FileNotFoundError:
        return "No Task Error"


def update_logs():
    global info
    try:
        return f"""{info.exp_result.logs}"""
    except FileNotFoundError:
        return f"""No Logs"""


def update_stats():
    global info
    try:
        stats = info.exp_result.steps_info[info.step].stats
        return pd.DataFrame(stats.items(), columns=["name", "value"])
    except (FileNotFoundError, IndexError):
        return None


def update_agent_info_md():
    global info
    try:
        agent_info = info.exp_result.steps_info[info.step].agent_info
        page = agent_info.get("markdown_page", None)
        if page is None:
            page = agent_info.get("markup_page", None)  # TODO: remove in a while
        if page is None:
            page = """Fill up markdown_page attribute in AgentInfo to display here."""
        return page
    except (FileNotFoundError, IndexError):
        return None


def update_agent_info_html():
    global info
    # screenshots from current and next step
    try:
        s1 = get_screenshot(info, info.step, False)
        s2 = get_screenshot(info, info.step + 1, False)
        agent_info = info.exp_result.steps_info[info.step].agent_info
        page = agent_info.get("html_page", ["No Agent Info"])
        if page is None:
            page = """Fill up html_page attribute in AgentInfo to display here."""
        else:
            page = _page_to_iframe(page)
        return page, s1, s2

    except (FileNotFoundError, IndexError):
        return None, None, None


def _page_to_iframe(page: str):
    html_bytes = page.encode("utf-8")
    encoded_html = base64.b64encode(html_bytes).decode("ascii")
    data_url = f"data:text/html;base64,{encoded_html}"

    # Create iframe with the data URL
    page = f"""
<iframe src="{data_url}" 
        style="width: 100%; height: 1000px; border: none; background-color: white;">
</iframe>
"""
    return page


def submit_action(input_text):
    global info
    agent_info = info.exp_result.steps_info[info.step].agent_info
    chat_messages = deepcopy(agent_info.get("chat_messages", ["No Chat Messages"])[:2])
    if isinstance(chat_messages[1], BaseMessage):  # TODO remove once langchain is deprecated
        assert isinstance(chat_messages[1], HumanMessage), "Second message should be user"
        chat_messages = [
            make_system_message(chat_messages[0].content),
            make_user_message(chat_messages[1].content),
        ]
    elif isinstance(chat_messages[1], dict):
        assert chat_messages[1].get("role", None) == "user", "Second message should be user"
    else:
        raise ValueError("Chat messages should be a list of BaseMessage or dict")

    client = OpenAI()
    chat_messages[1]["content"] = input_text
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_messages,
    )
    result_text = completion.choices[0].message.content
    return result_text


def update_prompt_tests():
    global info
    agent_info = info.exp_result.steps_info[info.step].agent_info
    chat_messages = agent_info.get("chat_messages", ["No Chat Messages"])
    prompt = chat_messages[1]
    if isinstance(prompt, dict):
        prompt = prompt.get("content", "No Content")
    return prompt, prompt


def select_step(episode_id: EpisodeId, evt: gr.SelectData):
    global info
    step = info.click_mapper.to_step(evt.index[0])
    info.step = step
    return StepId(episode_id, step)


def update_step_info():
    global info
    return [
        get_episode_info(info),
        get_action_info(info),
        get_state_error(info),
    ]


def get_obs(key: str, default=None):
    global info
    obs = info.exp_result.steps_info[info.step].obs
    return obs.get(key, default)


def code(txt):
    # return f"""<pre style="white-space: pre-wrap; word-wrap:
    # break-word;">{txt}</pre>"""
    return f"""```\n{txt}\n```"""


def get_episode_info(info: Info):
    try:
        env_args = info.exp_result.exp_args.env_args
        steps_info = info.exp_result.steps_info
        step_info = steps_info[info.step]
        try:
            goal = step_info.obs["goal_object"]
        except KeyError:
            goal = None
        try:
            cum_reward = info.exp_result.summary_info["cum_reward"]
        except FileNotFoundError:
            cum_reward = np.nan

        exp_dir = info.exp_result.exp_dir
        exp_dir_str = f"{exp_dir.parent.name}/{exp_dir.name}"

        info = f"""\
### {env_args.task_name} (seed: {env_args.task_seed})
### Step {info.step} / {len(steps_info)-1} (Reward: {cum_reward:.1f})

**Goal:**

{code(str(AgentLabBaseMessage('', goal)))}

**Task info:**

{code(step_info.task_info)}

**exp_dir:**

<small style="line-height: 1; margin: 0; padding: 0;">{code(exp_dir_str)}</small>"""
    except Exception as e:
        info = f"""\
**Error while getting episode info**
{code(traceback.format_exc())}"""
    return info


def get_action_info(info: Info):
    steps_info = info.exp_result.steps_info
    if len(steps_info) == 0:
        return "No steps were taken"
    if len(steps_info) <= info.step:
        return f"Step {info.step} is out of bounds. The episode has {len(steps_info)} steps."

    step_info = steps_info[info.step]
    action_info = f"""\
**Action:**

{code(step_info.action)}
"""
    think = step_info.agent_info.get("think", None)
    if think is not None:
        action_info += f"""
**Think:**

{code(think)}"""
    return action_info


def get_state_error(state: Info):
    try:
        step_info = state.exp_result.steps_info[state.step + 1]
        err_msg = step_info.obs.get("last_action_error", None)
    except (IndexError, AttributeError):
        err_msg = None

    if err_msg is None or len(err_msg) == 0:
        err_msg = "No Error"
    return f"""\
**Step error after action:**

{code(err_msg)}"""


def get_seeds_df(result_df: pd.DataFrame, task_name: str):
    result_df = result_df.reset_index(inplace=False)
    result_df = result_df[result_df[TASK_NAME_KEY] == task_name]

    def extract_columns(row: pd.Series):
        return pd.Series(
            {
                "seed": row[TASK_SEED_KEY],
                "reward": row.get("cum_reward", None),
                "err": bool(row.get("err_msg", None)),
                "n_steps": row.get("n_steps", None),
            }
        )

    seed_df = result_df.apply(extract_columns, axis=1)
    return seed_df


def on_select_agent(evt: gr.SelectData, df: pd.DataFrame):
    # TODO try to find a clever way to solve the sort bug here
    return info.get_agent_id(df.iloc[evt.index[0]])


def on_select_task(evt: gr.SelectData, df: pd.DataFrame, agent_id: list[tuple]):
    # get col index
    col_idx = df.columns.get_loc(TASK_NAME_KEY)
    return (agent_id, evt.row_value[col_idx])


def update_seeds(agent_task_id: tuple):
    agent_id, task_name = agent_task_id
    seed_df = get_seeds_df(info.agent_df, task_name)
    first_seed = seed_df.iloc[0]["seed"]
    return seed_df, EpisodeId(agent_id=agent_id, task_name=task_name, seed=first_seed)


def on_select_seed(evt: gr.SelectData, df: pd.DataFrame, agent_task_id: tuple):
    agent_id, task_name = agent_task_id
    col_idx = df.columns.get_loc("seed")
    seed = evt.row_value[col_idx]  # seed should be the first column
    return EpisodeId(agent_id=agent_id, task_name=task_name, seed=seed)


def new_episode(episode_id: EpisodeId, progress=gr.Progress()):
    print("new_episode", episode_id)
    global info
    info.update_exp_result(episode_id=episode_id)
    return generate_profiling(progress.tqdm), StepId(episode_id, info.step)


def fig_to_pil(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close(fig)
    return img_pil


def format_constant_and_variables():
    global info
    df = info.result_df
    constants, variables, _ = inspect_results.get_constants_and_variables(df)

    # map constants, a dict to a 2 column data frame with name and value
    constants = pd.DataFrame(constants.items(), columns=["name", "value"])
    records = []
    for var in variables:
        if var == "stack_trace":
            continue

        # get unique with count and sort by count descending
        unique_counts = df[var].value_counts().sort_values(ascending=False)

        for i, (val, count) in enumerate(unique_counts.items()):
            record = {
                "Name": var,
                "n unique": len(unique_counts),
                "i": i,
                "count": f"{count}/{len(df)}",
                "value": val,
            }

            records.append(record)
            if i >= 2:
                break

        if len(unique_counts) > 3:
            records.append(
                {
                    "Name": var,
                    "n unique": len(unique_counts),
                    "i": "...",
                    "count": "...",
                    "value": "...",
                }
            )
        records.append({"Name": ""})
    return constants, pd.DataFrame(records)


def get_agent_report(result_df: pd.DataFrame):
    levels = list(range(result_df.index.nlevels))

    if len(levels) == 1:
        result_df = result_df.set_index(AGENT_NAME_KEY, append=True)
        levels = list(range(result_df.index.nlevels))

    report = result_df.groupby(level=levels[1:]).apply(inspect_results.summarize)

    return report


def update_global_stats():
    stats = inspect_results.global_report(info.result_df, reduce_fn=inspect_results.summarize_stats)
    stats.reset_index(inplace=True)
    return stats


def update_error_report():
    report_files = list(info.exp_list_dir.glob("error_report*.md"))
    if len(report_files) == 0:
        return "No error report found"
    report_files = sorted(report_files, key=os.path.getctime, reverse=True)
    return report_files[0].read_text()


def new_exp_dir(exp_dir, progress=gr.Progress(), just_refresh=False):

    if exp_dir == select_dir_instructions:
        return None, None

    exp_dir = exp_dir.split(" - ")[0]

    if len(exp_dir) == 0:
        info.exp_list_dir = None
        return None, None

    info.exp_list_dir = info.results_dir / exp_dir
    info.result_df = inspect_results.load_result_df(info.exp_list_dir, progress_fn=progress.tqdm)
    info.result_df = remove_args_from_col(info.result_df)

    study_summary = inspect_results.summarize_study(info.result_df)
    # save study_summary
    study_summary.to_csv(info.exp_list_dir / "summary_df.csv", index=False)
    agent_report = display_table(study_summary)

    info.agent_id_keys = agent_report.index.names
    agent_report.reset_index(inplace=True)

    agent_id = info.get_agent_id(agent_report.iloc[0])

    constants, variables = format_constant_and_variables()
    return (
        agent_report,
        agent_id,
        constants,
        variables,
        update_global_stats(),
        update_error_report(),
    )


def new_agent_id(agent_id: list[tuple]):
    global info
    info.filter_agent_id(agent_id=agent_id)

    info.tasks_df = inspect_results.reduce_episodes(info.agent_df).reset_index()
    info.tasks_df = info.tasks_df.drop(columns=["std_err"])

    # task name of first element
    task_name = info.tasks_df.iloc[0][TASK_NAME_KEY]
    return info.tasks_df, (agent_id, task_name)


def get_directory_contents(results_dir: Path):
    exp_descriptions = []
    for dir in results_dir.iterdir():
        if not dir.is_dir():
            continue

        exp_description = dir.name
        try:
            # get summary*.csv files and find the most recent
            summary_files = list(dir.glob("summary*.csv"))
            if len(summary_files) != 0:
                most_recent_summary = max(summary_files, key=os.path.getctime)
                summary_df = pd.read_csv(most_recent_summary)

                # get row with max avg_reward
                max_reward_row = summary_df.loc[summary_df["avg_reward"].idxmax()]
                reward = max_reward_row["avg_reward"] * 100
                completed = max_reward_row["n_completed"]
                n_err = max_reward_row["n_err"]
                exp_description += (
                    f" - avg-reward: {reward:.1f}% - completed: {completed} - errors: {n_err}"
                )
        except Exception as e:
            print(f"Error while reading summary file: {e}")

        exp_descriptions.append(exp_description)

    return [select_dir_instructions] + sorted(exp_descriptions, reverse=True)


def most_recent_folder(results_dir: Path):
    return get_most_recent_study(results_dir).name


def refresh_exp_dir_choices(exp_dir_choice):
    global info
    return gr.Dropdown(
        choices=get_directory_contents(info.results_dir), value=exp_dir_choice, scale=1
    )


def generate_profiling(progress_fn):
    global info

    if info.exp_result is None:
        return None

    fig, ax = plt.subplots(figsize=(20, 3))

    try:
        summary_info = info.exp_result.summary_info
    except FileNotFoundError:
        summary_info = {}

    info.exp_result.progress_fn = progress_fn
    steps_info = info.exp_result.steps_info
    info.exp_result.progress_fn = None

    step_times = plot_profiling(ax, steps_info, summary_info, progress_fn)
    fig.tight_layout()
    info.click_mapper = ClickMapper(ax, step_times=step_times)

    return fig_to_pil(fig)


def add_patch(ax, start, stop, color, label, edge=False):
    if edge:
        ax.add_patch(
            patches.Rectangle(
                (start, 0),
                stop - start,
                1,
                edgecolor=color,
                alpha=1,
                label=label,
                fill=False,
                linewidth=3,
            )
        )
    else:
        ax.add_patch(
            patches.Rectangle((start, 0), stop - start, 1, color=color, alpha=1, label=label)
        )


def plot_profiling(ax, step_info_list: list[StepInfo], summary_info: dict, progress_fn):

    if len(step_info_list) == 0:
        warning("No step info to plot")
        return None

    # this allows to pop labels to make sure we don't use more than 1 for the legend
    labels = ["reset", "env", "agent", "exec action", "action error"]
    labels = {e: e for e in labels}

    colors = plt.get_cmap("tab20c").colors

    t0 = step_info_list[0].profiling.env_start
    all_times = []
    step_times = []
    for i, step_info in progress_fn(list(enumerate(step_info_list)), desc="Building plot."):
        step = step_info.step

        prof = deepcopy(step_info.profiling)
        # remove t0 from elements in profiling using for
        for key, value in prof.__dict__.items():
            if isinstance(value, float):
                setattr(prof, key, value - t0)
                all_times.append(value - t0)

        if i == 0:
            # reset
            add_patch(ax, prof.env_start, prof.env_stop, colors[14], labels.pop("reset", None))

        else:
            # env
            add_patch(ax, prof.env_start, prof.env_stop, colors[1], labels.pop("env", None))

            # action
            label = labels.pop("exec action", None)
            add_patch(ax, prof.action_exec_start, prof.action_exec_stop, colors[3], label)

            try:
                next_step_error = step_info_list[i + 1].obs["last_action_error"]
            except (IndexError, KeyError, TypeError):
                next_step_error = ""

            if next_step_error:
                # add a hollow rectangle for error
                label = labels.pop("action error", None)
                add_patch(ax, prof.env_start, prof.env_stop, "red", label, edge=True)

        if step_info.action is not None:
            # Blue rectangle for agent_start to agent_stop
            add_patch(ax, prof.agent_start, prof.agent_stop, colors[10], labels.pop("agent", None))

            # Black vertical bar at agent stop
            ax.axvline(prof.agent_stop, color="black", linewidth=3)
            step_times.append(prof.agent_stop)

            ax.text(
                prof.agent_stop,
                0,
                str(step + 1),
                color="white",
                fontsize=12,
                verticalalignment="bottom",
                horizontalalignment="left",
                rotation=0,
                clip_on=True,
                antialiased=True,
                fontweight=1000,
                backgroundcolor=colors[12],
            )

        if step_info.truncated or step_info.terminated:
            if step_info.truncated:
                color = "black"
            elif step_info.terminated:
                if summary_info.get("cum_reward", 0) > 0:
                    color = "limegreen"
                else:
                    color = "black"

            ax.axvline(prof.env_stop, color=color, linewidth=4, linestyle=":")

            text = f"R:{summary_info.get('cum_reward', np.nan):.1f}"

            if summary_info["err_msg"]:
                text = "Err"
                color = "red"

            ax.text(
                prof.env_stop,
                0.98,
                text,
                color="white",
                fontsize=12,
                verticalalignment="top",
                horizontalalignment="right",
                rotation=0,
                clip_on=True,
                antialiased=True,
                fontweight=1000,
                backgroundcolor=color,
            )

    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(all_times) + 1)
    # plt.gca().autoscale()

    ax.set_xlabel("Time")
    ax.set_yticks([])

    # position legend above outside the fig in one row
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=5,
        frameon=True,
    )

    return step_times


def main():
    run_gradio(RESULTS_DIR)


if __name__ == "__main__":
    main()
