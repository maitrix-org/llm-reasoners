import re
import json
from reasoners import SearchConfig, LanguageModel
from browsergym.core.action.highlevel import HighLevelActionSet
from utils.prompts import build_propose_prompt, build_evaluation_prompt
from utils.misc import check_validity_of_action_proposal

from gym_env import ActionGym, StateGym


class SearchConfigBrowsergym(SearchConfig):

    """
    SearchConfig for the browsergym environment.

    Attributes:
    - action_set (HighLevelActionSet): the action set for the browsergym environment
    - llm (LanguageModel): the language model used for generating proposals and evaluations
    - n_proposals (int): the number of proposals to generate
    - proposal_temperature (float): the temperature to use for generating proposals
    - evaluation_temperature (float): the temperature to use for generating evaluations
    - use_axtree (bool): whether to use the axtree in the prompts
    - use_html (bool): whether to use the page's html in the prompts
    - use_screenshot (bool): whether to use the screenshot (base64 encoded) in the prompts
    """

    def __init__(self,
                 action_set: HighLevelActionSet,
                 llm: LanguageModel,
                 n_proposals: int = 5, proposal_temperature: float = 0.7,
                 evaluation_temperature: float = 0.25,
                 use_axtree: bool = True, use_html: bool = False, use_screenshot: bool = False) -> None:
        super().__init__()
        self.action_set = action_set
        self.llm = llm
        self.n_proposals = n_proposals
        self.proposal_temperature = proposal_temperature
        self.evlaution_temperature = evaluation_temperature
        self.use_axtree = use_axtree
        self.use_html = use_html
        self.use_screenshot = use_screenshot

    def get_actions(self, state: StateGym) -> list[ActionGym]:
        """
        Generate a list of action proposals for the provided state. Proposals are generated by re-running the prompt at high temperature. Due to this, there can be duplicate proposals, so clustering is performed by looking at the final code that the proposal would run, and checking for uniqueness. Smaller LLMs (i.e. 4o-mini) often have issues with generating calls to functions that do not exist. Each function call in the proposal is checked to see if it is valid (check_validity_of_action_proposal).

        Args:
        - state (StateGym): the state to generate proposals for

        Returns:
        - clustered_actions (list[ActionGym]): a list of unique action proposals
        """

        system_msgs, user_msgs, full_prompt_text = build_propose_prompt(
            state.current_obs,
            self.action_set, state.action_history,
            self.use_axtree, self.use_html, self.use_screenshot
        )

        response = self.llm.generate(
            full_prompt_text, num_return_sequences=self.n_proposals, temperature=self.proposal_temperature)
        action_proposals = response.text

        clustered_actions = []
        action_codes = set()
        for action_proposal in action_proposals:
            if check_validity_of_action_proposal(action_proposal):
                action_code = self.action_set.to_python_code(action_proposal)
                if action_code not in action_codes:
                    action_codes.add(action_code)
                    clustered_actions.append(action_proposal)

        return clustered_actions

    def fast_reward(self, state: StateGym, action: ActionGym) -> tuple[float, dict]:
        """
        Generate an evaluation of a state action pair before using the action to step the environment. This process is entirely dependent on the LLM providing an accurate evaluation. The LLM provides a score from 0 to 10, which is then divided by 10 to keep the reward between 0 and 1 (important for UCT calculation in MCTS).

        Args:
        - state (StateGym): the state to evaluate
        - action (ActionGym): the action to evaluate

        Returns:
        - evaluation (float): the evaluation of the state action pair
        - aux (dict): used to pass the self-evaluation to the search algorithm, which then passes it to the SearchConfig's reward (not fast_reward) function
        """

        system_msgs, user_msgs, full_prompt_txt = build_evaluation_prompt(
            state.current_obs, action, self.action_set, state.action_history,
            self.use_axtree, self.use_html, self.use_screenshot
        )

        response = self.llm.generate(
            full_prompt_txt, num_return_sequences=self.n_proposals, temperature=self.proposal_temperature)

        evaluation = response.text[0]

        json_string = re.search(r"\{.*\}", evaluation, re.DOTALL).group()
        json_object = json.loads(json_string)
        evaluation = json_object["score"] / 10

        return evaluation, {"self_eval": evaluation}

    def reward(self, state: StateGym, action: ActionGym, **kwargs) -> tuple[float, dict]:
        """
        Generate a reward for a state action pair after stepping the environment with an action. The kwargs passed in are the combined aux dictionaries from the SearchConfig's fast_reward and EnvironmentGym's step functions. The env_reward for the browsergym environment is sparse, so a massive weight is provided to the environment's reward.
        """

        return kwargs["self_eval"] + 100 * kwargs["env_reward"], kwargs