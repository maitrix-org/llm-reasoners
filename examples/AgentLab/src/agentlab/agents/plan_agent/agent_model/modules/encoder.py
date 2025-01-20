from abc import abstractmethod
from typing import Dict
import logging


from ..base import AgentModule
from reasoners.lm.openai_model_w_parser import GenerateOutput

logger = logging.getLogger(__name__)


class BaseEncoder(AgentModule):
    def __init__(self, identity, *args, **kwargs):
        self.identity = identity

    @abstractmethod
    def __call__(self, observation, **kwargs): ...


class PromptedEncoder(BaseEncoder):
    def __init__(self, identity, llm, prompt_template, parser):
        super().__init__(identity)
        self.identity = identity
        self.llm = llm
        self.prompt_template = prompt_template
        self.parser = parser

    def __call__(
        self, observation_text, observation_screenshot, memory, **kwargs
    ) -> Dict[str, str]:
        user_prompt = self.prompt_template.format(
            observation=observation_text, memory=memory, **kwargs
        )
        llm_output: GenerateOutput = self.llm(
            system_prompt=str(self.identity),
            prompt=user_prompt,
            base64_image=observation_screenshot,
            parser=self.parser,
            **kwargs,
        )

        logger.debug(f"PromptedEncoder.__call__() `llm_output`: \n{llm_output.text}")
        return llm_output.text[0]
