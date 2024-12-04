import random
import time
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from browsergym.utils.obs import flatten_axtree_to_str

from ..base import AgentVariable


class BaseObservationSpace(AgentVariable):
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


class BrowserGymObservationSpace(BaseObservationSpace):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.goal = None
        self.error_accumulator = 0

    def get_value(self):
        return browser_observation_space_description

    def parse_observation(self, obs):
        scroll_position = obs['scroll_position']
        error_prefix = ''
        # since goal is specified within the environment, have to access it this way
        self.goal = obs['goal_object'][0]["text"]
        current_obs = {}
        if obs['last_action_error']:
            # add error recovery prompt prefix
            error_prefix = f'IMPORTANT! Last action is incorrect:\n{obs["last_action"]}\n{obs["last_action_error"]}\nThink again with the current observation of the page.\n'

        try:
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
            return current_obs, {'return_action': "send_msg_to_user('Error encountered when browsing.')"}

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

        scroll_progress = (
            1 - scroll_position['remainingPixels'] /
            scroll_position['documentHeight']
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
            'screenshot_som_base64': obs['screenshot_som_base64'],
        }

        if error_prefix:
            self.error_accumulator += 1
            if self.error_accumulator > 3:
                return current_obs, {'return_action': "send_msg_to_user('Too many errors encountered. Task failed.')"}
        else:
            self.error_accumulator = 0

        return current_obs, {}
