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
    ):
        super(Identity).__init__()
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.user_instruction = user_instruction
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        self.user_instruction = None

    def update(self, user_instruction):
        self.user_instruction = user_instruction

    def get_value(self):
        current_datetime = datetime.now().strftime('%a, %b %d, %Y %H:%M:%S')
        
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
{self.user_instruction}

# Current Date and Time:
{current_datetime}\
"""
