import ast
from abc import abstractmethod
import json
from ..base import AgentVariable


class ActionSpace(AgentVariable):
    @abstractmethod
    def parse_action(self, action, *args, **kwargs): ...


class BrowserGymActionSpace(ActionSpace): 
    def __init__(
        self,
        action_subsets=('chat', 'bid'),
        use_nav=True,
        strict=False,
        multiaction=False,
    ):
        super().__init__()

        self.action_subsets = action_subsets
        self.use_nav = use_nav
        self.strict = strict
        self.multiaction = multiaction

        if self.use_nav:
            self.action_subsets.append('nav')
        
        from browsergym.core.action.highlevel import HighLevelActionSet
        self.action_space = HighLevelActionSet(
            subsets=self.action_subsets,
            strict=self.strict,  # less strict on the parsing of the actions
            multiaction=self.multiaction,  # enable to agent to take multiple actions at once
        )
        
        self.reset()
        
    def reset(self):
        self.last_action = ''
        self.num_repeats = 0

    def get_value(self):
        return self.action_space.describe(
            with_long_description=False, with_examples=True
        )
        
    def parse_action(self, action, step_info, **kwargs):
        if not action.startswith('scroll') and action == self.last_action:
            self.num_repeats += 1
        else:
            self.num_repeats = 0
            self.last_action = action
            
        if self.num_repeats >= 3:
            action = 'send_msg_to_user("Repetitive actions. Ending the task.")'
            step_info.update({'action': action})
            
        return action, step_info


class EasyWebBrowserActionSpace(BrowserGymActionSpace):
    def parse_action(self, action, thought, **kwargs):
        from easyweb.events.action import (
            AgentFinishAction,
            BrowseInteractiveAction,
            MessageAction,
        )
        for action_type in [AgentFinishAction, BrowseInteractiveAction, MessageAction]:
            if isinstance(action, action_type):
                return action
            
        if isinstance(thought, dict): 
            thought = json.dumps(thought)
        action_str = action

        # handle send message to user function call in BrowserGym
        msg_content = ''
        for sub_action in action_str.split('\n'):
            if 'send_msg_to_user(' in sub_action:
                tree = ast.parse(sub_action)
                args = tree.body[0].value.args  # type: ignore
                msg_content = args[0].value

        return BrowseInteractiveAction(
            browser_actions=action_str,
            thought=thought,
            browsergym_send_msg_to_user=msg_content,
        )