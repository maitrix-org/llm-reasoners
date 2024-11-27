import json
from datetime import datetime
from functools import partial

from core.llms import OpenDevinParserLLM, OpenDevinParserMultiResponseLLM
from core.modules import (
    LLMReasonerPlanner,
    PolicyPlanner,
    PromptedActor,
    PromptedCritic,
    PromptedEncoder,
    StateMemoryUpdateEncoder,
    PromptedPolicy,
    PromptedWorldModel,
    KnowledgePromptedWorldModel,
)
from core.variables import (
    AgentInstructionEnvironmentIdentity,
    OpenDevinBrowserActionSpace,
    BrowserGymObservationSpace,
    StepKeyValueMemory,
    PromptedMemory,
    StepPromptedMemory,
)
from prompts import (
    actor_prompt_template_dict,
    critic_prompt_template,
    encoder_prompt_template_dict,
    encoder_memory_prompt_template,
    encoder_memory_prompt_template_dict,
    memory_prompt_template,
    policy_prompt_template_dict,
    world_model_prompt_template_dict,
)


from utils.llm import LLM
from utils.logger import AgentLogger
from utils.utils import ParseError, parse_html_tags_raise


def parser(text, keys, optional_keys=()):
    try:
        ans_dict = parse_html_tags_raise(text, keys, optional_keys)
    except ParseError as e:
        return None, False, str(e)
    return ans_dict, True, ''

class WebAgent():
    def __init__(self, 
                 llm: LLM,
                 use_llama: bool,
                 use_world_model_planning: bool,
                 #  use_intent_only_memory: bool,
                 #  use_prompted_memory: bool,
                 #  use_no_memory_actor: bool,
                 use_state_memory_encoder: bool,
                 memory_type: str,
                 encoder_prompt_type: str,
                 policy_prompt_type: str,
                 actor_prompt_type: str,
                 world_model_prompt_type: str,
                 planner_search_num_actions: int,
                 planner_search_depth: int,
                 planner_critic_num_samples: int,):
        self.llm = llm
        
        self.use_llama = use_llama
        self.use_world_model_planning = use_world_model_planning
        # self.use_prompted_memory = use_prompted_memory
        # self.use_no_memory_encoder = use_no_memory_encoder
        # self.use_intent_only_memory = use_intent_only_memory
        # self.use_no_memory_actor = use_no_memory_actor
        self.use_state_memory_encoder = use_state_memory_encoder
        self.memory_type = memory_type
        self.encoder_prompt_type = encoder_prompt_type
        self.policy_prompt_type = policy_prompt_type
        self.actor_prompt_type = actor_prompt_type
        self.world_model_prompt_type = world_model_prompt_type
        self.planner_search_num_actions = planner_search_num_actions
        self.planner_search_depth = planner_search_depth
        self.planner_critic_num_samples = planner_critic_num_samples
        
        # Action space and observation space
        self.action_space = OpenDevinBrowserActionSpace(
            action_subsets=['chat', 'bid'],
            use_nav=True,
            strict=False,
            multiaction=False,
        )
        self.observation_space = BrowserGymObservationSpace()

        # Agent identity
        agent_name = 'Web Browsing Agent'
        
        agent_description = 'An information and automation assistant who responds to \
user instructions by browsing the internet. The assistant strives to answer each question \
accurately, thoroughly, efficiently, and politely, and to be forthright when it is \
impossible to answer the question or carry out the instruction. The assistant will \
end the task once it sends a message to the user.'
        if self.use_llama:
            agent_description += ' The assistant remembers that bids are numbers in square brackets at the beginning of each line, and prioritizes reputable or stable websites like Google, Wikipedia, and Google Flights.'
        self.identity = AgentInstructionEnvironmentIdentity(
            agent_name=agent_name,
            agent_description=agent_description,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        
        # Encoder
        encoder_parser = partial(parser, keys=['state'])
        self.encoder_llm = OpenDevinParserLLM(llm, default_parser=encoder_parser)
        # TODO: Produce the prompt templates
        # if self.use_no_memory_encoder:
        #     encoder_prompt_template = encoder_prompt_template_dict['no_memory']
        # else:
        #     encoder_prompt_template = encoder_prompt_template_dict['with_memory']
        encoder_prompt_template = encoder_prompt_template_dict[self.encoder_prompt_type]
            
        if self.use_state_memory_encoder:
            memory_update_parser = partial(parser, keys=['memory_update'])
            self.memory_update_llm = OpenDevinParserLLM(llm, default_parser=memory_update_parser)
            self.encoder = StateMemoryUpdateEncoder(
                self.identity, self.encoder_llm, encoder_prompt_template, 
                self.memory_update_llm, encoder_memory_prompt_template
            )
        else:
            self.encoder = PromptedEncoder(
                self.identity, self.encoder_llm, prompt_template=encoder_prompt_template
            )
        
        # Memory
        # if self.use_prompted_memory:
        #     memory_parser = partial(parser, keys=['updated_memory'])
        #     self.memory_llm = OpenDevinParserLLM(llm, default_parser=memory_parser)
        #     self.memory = PromptedMemory(self.identity, self.memory_llm, prompt_template=memory_prompt_template)
        # else:
        #     if self.use_intent_only_memory:
        #         self.memory = StepKeyValueMemory(['intent'])
        #     else:
        #         self.memory = StepKeyValueMemory(['state', 'intent'])
                
        if self.memory_type == 'prompted':
            memory_parser = partial(parser, keys=['updated_memory'])
            self.memory_llm = OpenDevinParserLLM(llm, default_parser=memory_parser)
            self.memory = PromptedMemory(self.identity, self.memory_llm, prompt_template=memory_prompt_template)
        elif self.memory_type == 'step_prompted':
            memory_update_parser = partial(parser, keys=['memory_update'])
            self.memory_update_llm = OpenDevinParserLLM(llm, default_parser=memory_update_parser)
            if self.use_llama:
                memory_update_prompt_template = encoder_memory_prompt_template_dict['llama']
            else:
                memory_update_prompt_template = encoder_memory_prompt_template_dict['gpt-4o']
            self.memory = StepPromptedMemory(self.identity, self.memory_update_llm, 
                                             prompt_template=memory_update_prompt_template, 
                                             keys=['intent'])
        elif self.memory_type == 'step_key_value':
            self.memory = StepKeyValueMemory(['state', 'intent'])
        elif self.memory_type == 'step_key_value_intent_only':
            self.memory = StepKeyValueMemory(['intent'])
        else: 
            raise ValueError(f'Invalid memory type: {self.memory_type}')
        
        # Planner
        policy_prompt_template = policy_prompt_template_dict[self.policy_prompt_type]
        if self.use_world_model_planning:
            policy_parser = partial(parser, keys=['intent'], optional_keys=['think'])
            self.policy_llm = OpenDevinParserMultiResponseLLM(
                llm, default_parser=policy_parser
            )
            self.policy = PromptedPolicy(
                self.identity, self.policy_llm, prompt_template=policy_prompt_template
            )

            world_model_parser = partial(parser, keys=['next_state'])
            self.world_model_llm = OpenDevinParserLLM(
                llm, default_parser=world_model_parser
            )
            world_model_prompt_template = world_model_prompt_template_dict[self.world_model_prompt_type]
            
            if 'with_knolwedge' in world_model_prompt_template: 
                knowledge = open('../notebooks/episodic_memory_abstract.txt').read().strip()
                knowledge = """\
# Episodic Memory

Below is your abstract memory of past interactions. This memory will be used to guide you in predicting the consequences of your future actions.\
""" + '\n\n' + knowledge
                self.world_model = KnowledgePromptedWorldModel(
                    self.identity,
                    self.world_model_llm,
                    prompt_template=world_model_prompt_template,
                    knowledge=knowledge,
                )
            else:
                self.world_model = PromptedWorldModel(
                    self.identity,
                    self.world_model_llm,
                    prompt_template=world_model_prompt_template,
                )

            critic_parser = partial(
                parser, keys=['status', 'on_the_right_track'], optional_keys=['think']
            )
            self.critic_llm = OpenDevinParserMultiResponseLLM(
                llm, default_parser=critic_parser
            )
            self.critic = PromptedCritic(
                self.identity, self.critic_llm, prompt_template=critic_prompt_template
            )

            # self.planner = PolicyPlanner(self.policy)
            self.planner = LLMReasonerPlanner(self.policy, self.world_model, self.critic,
                                              search_num_actions=self.planner_search_num_actions,
                                              search_depth=self.planner_search_depth,
                                              critic_num_samples=self.planner_critic_num_samples,
                                              llm_base_url=llm.base_url,
                                              llm_api_key=llm.api_key)
            
        else:
            policy_parser = partial(parser, keys=['intent'], optional_keys=['think'])
            self.policy_llm = OpenDevinParserLLM(
                llm, default_parser=policy_parser
            )
            self.policy = PromptedPolicy(
                self.identity, self.policy_llm, prompt_template=policy_prompt_template
            )
            
            self.planner = PolicyPlanner(self.policy)

        # Actor
        action_parser = partial(parser, keys=['action'])
        self.actor_llm = OpenDevinParserLLM(llm, default_parser=action_parser)
        actor_prompt_template = actor_prompt_template_dict[self.actor_prompt_type]
        # if self.use_no_memory_actor:
        #     actor_prompt_template = actor_prompt_template_dict['no_memory']
        # else:
        #     actor_prompt_template = actor_prompt_template_dict['with_memory']
        self.actor = PromptedActor(
            self.identity, self.actor_llm, prompt_template=actor_prompt_template
        )

        self.reset()

    def reset(self):
        self.identity.reset()
        self.memory.reset()
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        log_file = f'{timestamp}.log'
        self.logger = AgentLogger(log_file)
        self.planner.logger = self.logger
        
        self.encoder_llm.logger = self.logger
        self.memory_update_llm.logger = self.logger
        self.policy_llm.logger = self.logger
        self.actor_llm.logger = self.logger
        if self.use_world_model_planning:
            self.world_model_llm.logger = self.logger
            self.critic_llm.logger = self.logger
        
        self.last_action = ''
        self.num_repeats = 0

    def step(self, raw_obs):
        observation, info = self.observation_space.parse_observation(raw_obs)
        if info.get('return_action') is not None:
            step = {
                'observation': observation,
                'state': None,
                'intent': None,
                'action': info['return_action'],
            }
            return info['return_action'], step
        self.identity.update(user_instruction=observation['goal'])

        obs_txt = observation['clean_axtree_txt']
        # logger.info(f'*Observation*: {obs_txt}')
        self.logger.info(f'*Observation*: {obs_txt}')

        kwargs = {}
        if self.use_state_memory_encoder:
            llm_output = self.encoder(obs_txt, self.memory)
            state, memory_update = llm_output['state'], llm_output['memory_update']
            self.logger.info(f'*State*: {state}')
            self.logger.info(f'*Memory update*: {memory_update}')
            kwargs['memory_update'] = memory_update
        else:
            # state = self.encoder(obs_txt, self.memory)['state']
            state = self.encoder(obs_txt, self.memory).get('state')
            self.logger.info(f'*State*: {state}')
        if not state: 
            return_action = 'send_msg_to_user("LLM output parsing error")'
            step = {
                'observation': observation,
                'state': None,
                'intent': None,
                'action': return_action,
            }
            return return_action, step

        # intent = self.planner(state, self.memory, **kwargs)['intent']
        intent = self.planner(state, self.memory, **kwargs).get('intent')
        self.logger.info(f'*Intent*: {intent}')
        if not intent:
            return_action = 'send_msg_to_user("LLM output parsing error")'
            step = {
                'observation': observation,
                'state': state,
                'intent': None,
                'action': return_action,
            }
            return return_action, step

        # action = self.actor(obs_txt, state, self.memory, intent, **kwargs)['action']
        action = self.actor(obs_txt, state, self.memory, intent, **kwargs).get('action')
        self.logger.info(f'*Action*: {action}')
        if not action:
            return_action = 'send_msg_to_user("LLM output parsing error")'
            step = {
                'observation': observation,
                'state': state,
                'intent': intent,
                'action': return_action,
            }
            return return_action, step

        if self.use_state_memory_encoder:
            step = {
                'observation': observation,
                'state': memory_update,
                'state_original': state,
                'intent': intent,
                'action': action,
            }
        else:
            step = {
                'observation': observation,
                'state': state,
                'intent': intent,
                'action': action,
            }
        
        try:
            self.memory.update(**step)
        except KeyError as e: 
            self.logger.info(f'*Memory update error*: {e}')
            return_action = 'send_msg_to_user("LLM output parsing error")'
            return return_action, step
        
        step.update(self.memory.current_step)
        if self.memory_type == 'step_prompted':
            self.logger.info(f"*Memory update*: {self.memory.current_step['memory_update']}")
        # self.logger.info(f'*Memory*: {self.memory}')
        
        self.memory.step()
        
        if not action.startswith('scroll') and action == self.last_action:
            self.num_repeats += 1
        else:
            self.num_repeats = 0
            self.last_action = action
            
        total_cost = 0
        for llm in [self.encoder_llm, self.memory_update_llm, 
                    self.policy_llm, self.actor_llm]:
            total_cost += llm.cost_accumulator
        if self.use_world_model_planning:
            total_cost += self.world_model_llm.cost_accumulator
            total_cost += self.critic_llm.cost_accumulator
        self.logger.info(f'*Total Accumulated Cost*: {total_cost:.2f}')
            
        if self.num_repeats >= 3:
            action = 'send_msg_to_user("Repetitive actions. Ending the task.")'
            step['action'] = action
            
        
        # return self.action_space.parse_action(action, thought=json.dumps(step))
        return action, step
