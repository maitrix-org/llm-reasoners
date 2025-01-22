import base64
from io import BytesIO
import json
from glob import glob

import gradio as gr
from PIL import Image
# from session import TestSession

MAX_TABS = 30


def load_history(log_selection):
    data = json.load(open(log_selection, 'r'))
    user_message = data['goal']
    error_message = data.get('error', '')
    steps = []
    for step in data['history']:
        image_data = base64.b64decode(step[0]['screenshot'])
        screenshot = Image.open(BytesIO(image_data))
        step_data = {
            'url': step[0]['url'],
            'screenshot': screenshot,
            'observation': step[2]['obs_info']['clean_axtree_txt'],
            'state': step[2]['state'],
            'plan': step[2]['plan'],
            'action': step[2]['action'],
        }
        steps.append(step_data)
    
    actions = [step[1] for step in data['history']] + ['Error: ' + error_message]
    chat_history = [[user_message, '\n\n'.join(actions)]]
    
    tabs = []
    start_url = 'about:blank'
    blank = Image.new('RGB', (1280, 720), (255, 255, 255))
    placeholder = '<placeholder>'

    # print(self.browser_history)
    tabs = []
    urls = []
    screenshots = []
    sub_tabs = []
    
    observations = []
    states = []
    plans = []
    actions = []

    for i in range(MAX_TABS):
        step_data = steps[i] if i < len(steps) else None
        visible = i < len(data['history'])
        with gr.Tab(f'Step {i + 1}', visible=visible) as tab:
            with gr.Group():
                browser_step = (
                    (step_data['screenshot'], step_data['url'])
                    if step_data
                    else (blank, start_url)
                )
                # content = urls[i] if i < len(urls) else start_url
                url = gr.Textbox(
                    browser_step[1], label='URL', interactive=False, max_lines=1
                )
                # content = screenshots[i] if i < len(screenshots) else blank
                screenshot = gr.Image(
                    browser_step[0], interactive=False, label='Webpage'
                )
                urls.append(url)
                screenshots.append(screenshot)

                with gr.Tab('Observation') as obs_tab:
                    content = (
                        step_data['observation']
                        if step_data
                        else placeholder
                    )
                    observation = gr.Textbox(
                        content,
                        interactive=False,
                        lines=20,
                        max_lines=30,
                    )
                    observations.append(observation)
                    sub_tabs.append(obs_tab)
                with gr.Tab('State') as state_tab:
                    content = step_data['state'] if step_data else placeholder
                    state = gr.Textbox(
                        content,
                        interactive=False,
                        lines=20,
                        max_lines=30,
                    )
                    states.append(state)
                    sub_tabs.append(state_tab)
                with gr.Tab('Plan') as plan_tab:
                    content = (
                        step_data['plan']
                        if step_data
                        else placeholder
                    )
                    plan = gr.Textbox(
                        content,
                        interactive=False,
                        lines=20,
                        max_lines=30,
                    )
                    plans.append(plan)
                    sub_tabs.append(plan_tab)
                with gr.Tab('Action') as action_tab:
                    content = step_data['action'] if step_data else placeholder
                    action = gr.Textbox(
                        content,
                        interactive=False,
                        lines=20,
                        max_lines=30,
                    )
                    actions.append(action)
                    sub_tabs.append(action_tab)

            tabs.append(tab)

    # print(len(tabs))
    return (
        [chat_history]
        + tabs
        + urls
        + screenshots
        + sub_tabs
        + observations
        + states
        + plans
        + actions
    )


def select_log_dir(log_dir_selection):
    log_list = list(reversed(sorted(glob(f'./{log_dir_selection}/*.json'))))
    return gr.Dropdown(
        log_list,
        value=None,
        interactive=True,
        label='Log',
        info='Choose the log to visualize',
    )


def refresh_log_selection(log_dir_selection):
    log_list = list(reversed(sorted(glob(f'./{log_dir_selection}/*.json'))))
    return gr.Dropdown(
        log_list,
        value=None,
        interactive=True,
        label='Log',
        info='Choose the log to visualize',
    )
