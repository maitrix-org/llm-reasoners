from abc import abstractmethod
from ..base import AgentVariable

from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import time, random

if TYPE_CHECKING:
    from easyweb.controller.state.state import State as EasyWebState


class ObservationSpace(AgentVariable):
    @abstractmethod
    def parse_observation(self, *args, **kwargs) -> Tuple[Any, Dict]: ...
    

browser_observation_space_description = """\
The text representation and screenshot of the part of webpage visible in the viewport of a browser. \
Here is an abstract description of the information available in the webpage text representation:

- Identification Information:
    - URL: The web address that specifies the location of the webpage.
    - Document Properties: Attributes such as scroll position and viewport dimensions that describe the current viewing context.

- Structural Hierarchy:
    - Root Element: The primary container for the webpage, indicating its overall theme or purpose.
    - Nested Elements: A hierarchy of sections, containers, and components that organize content logically (e.g., headers, footers, sidebars).

- Interactive Components:
    - Links: Elements that can be clicked to navigate to other pages or sections, often labeled descriptively.
    - Buttons: Interactive controls that trigger actions (e.g., submitting forms, opening menus).

- Content Types:
    - Text: Main content, headings, and subheadings that provide information and context.
    - Images and Media: Visual elements that enhance the understanding or appeal of the content.
    - Forms and Inputs: Fields for user input, including text boxes, dropdowns, and checkboxes.

- Functional Areas:
    - Navigation Menus: Organized sets of links that allow users to explore different sections of the site.
    - Search Interface: Components that enable users to search for content within the site, including input fields and associated buttons.

- State Information:
    - Visibility and Expand/Collapse States: Indicators showing whether certain elements are active, visible, or in a collapsed state, impacting user interaction.
    - Focus States: Information on which elements are currently focused, important for keyboard navigation and accessibility.

- Accessibility Features:
    - Role and Description Information: Metadata that provides context about the purpose of elements, useful for screen readers and assistive technologies.

-  General User Considerations:
    - Navigation: Recognizing how to traverse the webpage using links and buttons.
    - Interactivity: Understanding how to engage with forms, search fields, and dynamic components.
    - Content Engagement: Identifying and interpreting various content types to glean necessary information.\
"""

class BrowserGymObservationSpace(ObservationSpace):
    def __init__(self, truncation=True):
        super().__init__()
        self.reset()
        self.truncation = truncation
        
    def reset(self):
        self.goal = None
        self.error_accumulator = 0
        
    def get_value(self):
        return browser_observation_space_description
        
    def parse_observation(self, obs):
        scroll_position = obs['scroll_position']
        error_prefix = ''
        self.goal = obs['goal']
        current_obs = {}
        
        if obs['last_action_error']:
            # add error recovery prompt prefix
            error_prefix = f'IMPORTANT! Last action is incorrect:\n{obs["last_action"]}\n{obs["last_action_error"]}\nThink again with the current observation of the page.\n'

        try:
            from browsergym.utils.obs import flatten_axtree_to_str
            cur_axtree_txt = flatten_axtree_to_str(
                obs['axtree_object'],
                extra_properties=obs['extra_element_properties'],
                with_clickable=True,
                filter_visible_only=True,
            )
        except Exception as e:
            print(
                'Error when trying to process the accessibility tree: %s', e
            )
            # cur_axtree_txt = 'Error when trying to process the accessibility tree. No observation is available.'
            return None, {'return_action': "send_msg_to_user('Error encountered when browsing.')"}
        
        if self.truncation:
            clean_axtree_lines = []
            num_static_text_lines = 0
            max_static_text_lines = 20
            last_bracket_line = 0
            max_after_last_bracket_lines = 10
            for i, line in enumerate(cur_axtree_txt.split('\n')):
                if line.strip().startswith('['):
                    last_bracket_line = i

            for i, line in enumerate(cur_axtree_txt.split('\n')):
                if line.strip().startswith('StaticText') or line.strip().startswith(
                    'ListMarker'
                ):
                    num_static_text_lines += 1
                else:
                    num_static_text_lines = 0

                if num_static_text_lines <= max_static_text_lines and i < (
                    last_bracket_line + max_after_last_bracket_lines
                ):
                    clean_axtree_lines.append(line)

            clean_axtree_txt = '\n'.join(clean_axtree_lines)
        else:
            clean_axtree_txt = cur_axtree_txt

        scroll_progress = (
            1 - scroll_position['remainingPixels'] / scroll_position['documentHeight']
        )
        clean_axtree_txt = (
            f"URL {obs['url']}\n"
            f"Scroll Position: {scroll_position['scrollTop']}, "
            f"Window Height: {scroll_position['windowHeight']}, "
            f"Webpage Height: {scroll_position['documentHeight']}, "
            f"Remaining Pixels: {scroll_position['remainingPixels']}, "
            f"Scrolling Progress: {scroll_progress:.1%}\n"
        ) + clean_axtree_txt

        obs_prompt = clean_axtree_txt
        if len(error_prefix) > 0:
            obs_prompt = f'{error_prefix}\n' + obs_prompt
        
        current_obs = {
            'clean_axtree_txt': obs_prompt,
            'error_prefix': error_prefix,
            'goal': self.goal,
        }
        obs_txt = obs_prompt
        obs_info = current_obs
        
        if error_prefix:
            self.error_accumulator += 1
            if self.error_accumulator > 3:
                obs_info.update({'return_action': "send_msg_to_user('Too many errors encountered. Task failed.')"})
                return obs_txt, obs_info
        else:
            self.error_accumulator = 0
            

        # return current_obs, {}
        return obs_txt, obs_info
    
    
class EasyWebBrowserObservationSpace(BrowserGymObservationSpace):
    """An identity that describes the agent and the environment it is in."""

    def __init__(self, eval_mode, truncation=True):
        super().__init__()
        self.eval_mode = eval_mode
        self.truncation = truncation
        self.reset()

    def parse_observation(self, easyweb_state: "EasyWebState") -> Tuple[Any, Dict]:
        last_obs, last_action, return_action = self._process_control_flow(
            easyweb_state
        )
        if return_action is not None:
            return None, {'return_action': return_action}

        # current_obs, return_action = self._parse_current_obs(last_obs)
        obs_info, return_action = self._parse_current_obs(last_obs)
        obs_txt = obs_info.get('clean_axtree_txt')
        if return_action:
            obs_info.update({'return_action': return_action})
        # return current_obs, {'return_action': return_action}
        return obs_txt, obs_info

    def _process_control_flow(self, env_state):
        from easyweb.events.action import (
            AgentFinishAction,
            BrowseInteractiveAction,
            MessageAction,
        )
        from easyweb.events.event import EventSource
        
        goal = env_state.get_current_user_intent()
        if goal is None:
            goal = env_state.inputs['task']
        self.goal = goal

        # messages: List[str] = []
        prev_actions: List[str] = []
        last_obs = None
        last_action = None

        # if EVAL_MODE and len(env_state.history) == 1:
        if len(env_state.history) == 1:
            # for webarena and miniwob++ eval, we need to retrieve the initial observation already in browser env
            # initialize and retrieve the first observation by issuing an noop OP
            # For non-benchmark browsing, the browser env starts with a blank page, and the agent is expected to first navigate to desired websites
            time.sleep(10 + random.random() * 5)
            return (
                last_obs,
                last_action,
                BrowseInteractiveAction(browser_actions='noop()'),
            )

        for prev_action, obs in env_state.history:
            # Go through the history to get the last action
            if isinstance(prev_action, BrowseInteractiveAction):
                # Create a list of past actions
                prev_actions.append(prev_action.browser_actions)
                last_obs = obs
                last_action = prev_action
            elif (
                isinstance(prev_action, MessageAction)
                and prev_action.source == EventSource.AGENT
            ):
                # agent has responded, task finish.
                return (
                    last_obs,
                    last_action,
                    AgentFinishAction(outputs={'content': prev_action.content}),
                )

        if self.eval_mode:
            prev_actions = prev_actions[1:]  # remove the first noop action

        # prev_action_str = '\n'.join(prev_actions)
        # if the final BrowserInteractiveAction exec BrowserGym's send_msg_to_user,
        # we should also send a message back to the user in EasyWeb and call it a day
        if (
            isinstance(last_action, BrowseInteractiveAction)
            and last_action.browsergym_send_msg_to_user
        ):
            # Here the browser interaction action from BrowserGym can also include a message to the user
            # When we see this browsergym action we should use a MessageAction from EasyWeb
            return (
                last_obs,
                last_action,
                MessageAction(last_action.browsergym_send_msg_to_user),
            )

        return last_obs, last_action, None

    def _parse_current_obs(self, last_obs):
        from browsergym.utils.obs import flatten_axtree_to_str
        from easyweb.events.observation import BrowserOutputObservation
        from easyweb.core.logger import easyweb_logger as logger
        from easyweb.events.action import MessageAction
        
        cur_axtree_txt = ''
        error_prefix = ''
        current_obs = {}

        if isinstance(last_obs, BrowserOutputObservation):
            # The browser output observation belongs to EasyWeb
            if last_obs.error:
                # add error recovery prompt prefix
                error_prefix += f'IMPORTANT! Last action is incorrect:\n{last_obs.last_browser_action}\n{last_obs.last_browser_action_error}\nThink again with the current observation of the page.\n'
            try:
                cur_axtree_txt = flatten_axtree_to_str(
                    last_obs.axtree_object,
                    extra_properties=last_obs.extra_element_properties,
                    with_clickable=True,
                    filter_visible_only=True,
                )
                scroll_progress = (
                    1
                    - last_obs.scroll_position['remainingPixels']
                    / last_obs.scroll_position['documentHeight']
                )
                cur_axtree_txt = (
                    f"URL {last_obs.url}\n"
                    f"Scroll Position: {last_obs.scroll_position['scrollTop']}, "
                    f"Window Height: {last_obs.scroll_position['windowHeight']}, "
                    f"Webpage Height: {last_obs.scroll_position['documentHeight']}, "
                    f"Remaining Pixels: {last_obs.scroll_position['remainingPixels']}, "
                    f"Scrolling Progress: {scroll_progress:.1%}\n"
                ) + cur_axtree_txt
                logger.info(last_obs.scroll_position)
            except Exception as e:
                logger.error(
                    'Error when trying to process the accessibility tree: %s', e
                )
                cur_axtree_txt = 'Error when trying to process the accessibility tree. No observation is available.'
                return current_obs, MessageAction('Error encountered when browsing.')

        if error_prefix:
            self.error_accumulator += 1
            if self.error_accumulator > 10:
                return current_obs, MessageAction(
                    'Too many errors encountered. Task failed.'
                )
        else:
            self.error_accumulator = 0

        ### Above is record keeping by world model

        if self.truncation:
            clean_axtree_lines = []
            num_static_text_lines = 0
            max_static_text_lines = 20
            last_bracket_line = 0
            max_after_last_bracket_lines = 10
            for i, line in enumerate(cur_axtree_txt.split('\n')):
                if line.strip().startswith('['):
                    last_bracket_line = i

            for i, line in enumerate(cur_axtree_txt.split('\n')):
                if line.strip().startswith('StaticText') or line.strip().startswith(
                    'ListMarker'
                ):
                    num_static_text_lines += 1
                else:
                    num_static_text_lines = 0

                if num_static_text_lines <= max_static_text_lines and i < (
                    last_bracket_line + max_after_last_bracket_lines
                ):
                    clean_axtree_lines.append(line)

            clean_axtree_txt = '\n'.join(clean_axtree_lines)

            obs_prompt = clean_axtree_txt
        else:
            obs_prompt = cur_axtree_txt

        if len(error_prefix) > 0:
            obs_prompt = f'{error_prefix}\n' + obs_prompt

        current_obs = {
            'clean_axtree_txt': obs_prompt,
            'raw_axtree_txt': cur_axtree_txt,
            'error_prefix': error_prefix,
            'goal': self.goal,
        }
        return current_obs, None