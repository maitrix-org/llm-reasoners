from glob import glob
import os

import gradio as gr
from controller import MAX_TABS, load_history, refresh_log_selection, select_log_dir
from PIL import Image

current_dir = os.path.dirname(__file__) 

with gr.Blocks() as demo:
    title = gr.Markdown('# ReasonerAgent-Web Log Visualizer')
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group():
                log_dir_options = [
                    'browsing_data',
                ]
                log_dir_options.extend(glob('**/*_logs/', recursive=True))
                default_logdir = log_dir_options[0]
                # print(f'./{default_logdir}/*.json')
                log_list = list(
                    reversed(sorted(glob(os.path.join(current_dir, '..', default_logdir, '*.json'))))
                )
                # print(log_list)
                log_dir_selection = gr.Dropdown(
                    log_dir_options, value=default_logdir, label='Log Directory'
                )
                log_selection = gr.Dropdown(
                    log_list,
                    value=None,
                    interactive=True,
                    label='Log',
                    info='Choose the log to visualize',
                )
                chatbot = gr.Chatbot()
            refresh = gr.Button('Refresh Log List')

        with gr.Column(scale=2):
            start_url = 'about:blank'
            blank = Image.new('RGB', (1280, 720), (255, 255, 255))
            placeholder = '<placeholder>'

            tabs = []
            urls = []
            screenshots = []
            sub_tabs = []
            observations = []
            states = []
            plans = []
            actions = []
            # plots = []
            # webpages = []
            for i in range(MAX_TABS):
                with gr.Tab(f'Step {i + 1}', visible=(i == 0)) as tab:
                    with gr.Group():
                        url = gr.Textbox(
                            start_url, label='URL', interactive=False, max_lines=1
                        )
                        screenshot = gr.Image(blank, interactive=False, label='Webpage')
                        with gr.Tab('Observation') as obs_tab:
                            observation = gr.Textbox(
                                placeholder,
                                interactive=False,
                                lines=20,
                                max_lines=30,
                            )
                            observations.append(observation)
                            sub_tabs.append(obs_tab)
                        with gr.Tab('State') as state_tab:
                            state = gr.Textbox(
                                placeholder,
                                interactive=False,
                                lines=20,
                                max_lines=30,
                            )
                            states.append(state)
                            sub_tabs.append(state_tab)
                        with gr.Tab('Plan') as plan_tab:
                            plan = gr.Textbox(
                                placeholder,
                                interactive=False,
                                lines=20,
                                max_lines=30,
                            )
                            plans.append(plan)
                            sub_tabs.append(plan_tab)
                        with gr.Tab('Action') as action_tab:
                            action = gr.Textbox(
                                placeholder,
                                interactive=False,
                                lines=20,
                                max_lines=30,
                            )
                            actions.append(action)
                            sub_tabs.append(action_tab)

                        urls.append(url)
                        screenshots.append(screenshot)

                    tabs.append(tab)
            # print(len(tabs))

    log_dir_selection.select(select_log_dir, log_dir_selection, log_selection)
    log_selection.select(
        load_history,
        log_selection,
        [chatbot]
        + tabs
        + urls
        + screenshots
        + sub_tabs
        + observations
        + states
        + plans
        + actions,
    )
    refresh.click(refresh_log_selection, log_dir_selection, log_selection)

if __name__ == '__main__':
    # log_list = list(reversed(sorted(glob('./frontend_logs/*.json'))))
    demo.queue()
    demo.launch(share=True)
