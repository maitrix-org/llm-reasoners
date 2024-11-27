import ast
from abc import abstractmethod

from browsergym.core.action.highlevel import HighLevelActionSet

# from opendevin.events.action import BrowseInteractiveAction

from ..base import AgentVariable


class BaseActionSpace(AgentVariable):
    @abstractmethod
    def parse_action(self, *args, **kwargs): ...


class OpenDevinBrowserActionSpace(BaseActionSpace):
    """An identity that describes the agent and the environment it is in."""

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
        self.action_space = HighLevelActionSet(
            subsets=self.action_subsets,
            strict=self.strict,  # less strict on the parsing of the actions
            multiaction=self.multiaction,  # enable to agent to take multiple actions at once
        )

    def get_value(self):
        return self.action_space.describe(
            with_long_description=False, with_examples=True
        )

    def parse_action(self, action, thought):
        # thought = ''
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
