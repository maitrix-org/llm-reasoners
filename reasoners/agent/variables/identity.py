from datetime import datetime
from ..base import AgentVariable


class Identity(AgentVariable): ...


class AgentInstructionEnvironmentIdentity(Identity):
    """An identity that describes the agent and the environment it is in."""

    def __init__(
        self,
        agent_name,
        agent_description,
        observation_space,
        action_space,
        user_instruction=None,
        with_datetime=True,
    ):
        super(Identity).__init__()
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.user_instruction = user_instruction
        self.observation_space = observation_space
        self.action_space = action_space
        self.with_datetime = with_datetime

    def reset(self):
        self.user_instruction = None

    def update(self, user_instruction):
        self.user_instruction = user_instruction

    def get_value(self):
        if self.with_datetime:
            current_datetime = datetime.now().strftime('%a, %b %d, %Y %H:%M:%S')
            datetime_string = f'\n\n# Current Date and Time:\n{current_datetime}'
        else:
            datetime_string = ''
        
        return f"""\
# Name:
{self.agent_name}

# Description:
{self.agent_description}

# Observation Space:
{self.observation_space}

# Action Space:
{self.action_space}

# Instruction:
{self.user_instruction}\
{datetime_string}\
"""
