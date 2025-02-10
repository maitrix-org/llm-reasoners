from typing import Dict
import logging

from ..base import AgentModule

from reasoners.base import GenerateOutput

logger = logging.getLogger(__name__)


class BasePolicy(AgentModule):
    def __init__(self, identity, *args, **kwargs):
        self.identity = identity


class PromptedPolicy(BasePolicy):
    def __init__(self, identity, llm, prompt_template, parser):
        super().__init__(identity)
        self.llm = llm
        self.prompt_template = prompt_template
        self.parser = parser

    def __call__(self, state, memory, llm_kwargs=None, **kwargs) -> list[Dict[str, str]]:
        if llm_kwargs is None:
            llm_kwargs = {}
        user_prompt = self.prompt_template.format(state=state, memory=memory, **kwargs)
        llm_output: GenerateOutput = self.llm(
            system_prompt=str(self.identity),
            prompt=user_prompt,
            parser=self.parser,
            **llm_kwargs,
        )

        logger.debug(f"PromptedPolicy.__call__() `system_prompt`: \n{str(self.identity)}")
        logger.debug(f"PromptedPolicy.__call__() `user_prompt`: \n{user_prompt}")
        logger.debug(f"PromptedPolicy.__call__() `llm_output`: \n{llm_output.text}")

        return llm_output.text
