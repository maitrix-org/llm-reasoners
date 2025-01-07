import base64
import json
from io import BytesIO

from PIL import Image, UnidentifiedImageError


class TestSession:
    def __init__(self):
        self.token = self.status = self.agent_state = self.figure = None
        self.action_messages = []
        self.browser_history = []
        # self.figures = []
        self.observations = []
        self.states = []
        self.plans = []
        self.actions = []

    def _read_message(self, message, verbose=True):
        printable = {}
        if message.get('token'):
            self.token = message['token']
            self.status = message['status']
            printable = message
        elif message.get('observation') == 'agent_state_changed':
            self.agent_state = message['extras']['agent_state']
            printable = message
        elif 'action' in message:
            self.action_messages.append(message['message'])
            if message['action'] == 'browse_interactive':
                obs, state, plan, action = self._load_browse_interactive(message)
                self.observations.append(obs)
                self.states.append(state)
                self.plans.append(plan)
                self.actions.append(action)
            printable = message
        elif 'extras' in message and 'screenshot' in message['extras']:
            image_data = base64.b64decode(message['extras']['screenshot'])
            try:
                screenshot = Image.open(BytesIO(image_data))
                url = message['extras']['url']
                printable = {
                    k: v for k, v in message.items() if k not in ['extras', 'content']
                }
                self.browser_history.append((screenshot, url))
            except UnidentifiedImageError:
                err_msg = (
                    'Failure to receive screenshot, likely due to a server-side error.'
                )
                self.action_messages.append(err_msg)
        if verbose:
            print(printable)

    def _load_browse_interactive(self, message):
        obs = state = plan = action = ''
        if (
            ('args' in message)
            and ('thought' in message['args'])
            and message['args']['thought'].startswith('{')
        ):
            print('Load webpage')
            agent_output = json.loads(message['args']['thought'])

            obs = agent_output.get('obs', {}).get('clean_axtree_txt', '')
            if not obs:
                obs = agent_output.get('observation', {}).get('clean_axtree_txt', '')
            state = agent_output.get('state', '')
            plan = agent_output.get('instruction', '')
            if not plan:
                plan = agent_output.get('intent', '')
            if not plan:
                plan = agent_output.get('plan', '')
            action = agent_output.get('action', '')

        return obs, state, plan, action
