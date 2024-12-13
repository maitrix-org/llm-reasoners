import copy

default_web_agent_config = {
    'agent_name': 'Web Browsing Agent',
    'agent_description': """An information and automation assistant who responds to \
user instructions by browsing the internet. The assistant strives to answer each question \
accurately, thoroughly, efficiently, and politely, and to be forthright when it is \
impossible to answer the question or carry out the instruction. The assistant will \
end the task once it sends a message to the user.""",
    'memory_type': 'step_prompted',
    'memory_prompt_type': 'default',
    'encoder_prompt_type': 'no_memory',
    'planner_type': 'policy',
    'policy_prompt_type': 'no_update',
    'policy_output_name': 'intent',
    'actor_prompt_type': 'with_memory',
    'module_error_message': 'send_msg_to_user("LLM output parsing error")'
}

browsergym_config = copy.copy(default_web_agent_config)
browsergym_config.update({
    'environment': 'browsergym',
})

browsergym_world_model_config = copy.copy(browsergym_config)
browsergym_world_model_config.update({
    'planner_type': 'world_model',
    'world_model_prompt_type': 'with_update',
    'planner_search_num_actions': 5,
    'planner_search_depth': 1,
    'planner_critic_num_samples': 20,
})

opendevin_config = copy.copy(default_web_agent_config)
opendevin_config.update({
    'environment': 'opendevin',
    'use_nav': True,
    'eval_mode': False,
    'truncate_axtree': True,
})

opendevin_llama_config = copy.copy(opendevin_config)
opendevin_llama_config.update({
    'agent_description': 'An information and automation assistant who responds to \
user instructions by browsing the internet. The assistant strives to answer each question \
accurately, thoroughly, efficiently, and politely, and to be forthright when it is \
impossible to answer the question or carry out the instruction. The assistant will \
end the task once it sends a message to the user. The assistant remembers that bids \
are numbers in square brackets at the beginning of each line, and prioritizes reputable \
or stable websites like Google, Wikipedia, and Google Flights.',
    'memory_prompt_type': 'llama'
})

opendevin_world_model_config = copy.copy(opendevin_config)
opendevin_world_model_config.update({
    'planner_type': 'world_model',
    'world_model_prompt_type': 'with_update',
    'planner_search_num_actions': 5,
    'planner_search_depth': 1,
    'planner_critic_num_samples': 20,
})

webarena_config = copy.copy(opendevin_config)
webarena_config.update({
    'agent_description': """An information and automation assistant that interacts with the browser \
and responds to user instructions. The response follows the following rules: \
1. When the intent is a question, and a complete answer to the question has been found, \
then send the answer to the user; 2. the intent wants to locate specific information or navigate to \
a particular section of a site, and the current page satisfies, then stop and tell the user you found the required information; \
3. the intent want to conduct an operation, and has been done, then stop and tell the user the operation has been completed."
The assistatnt should try to acheive the goal in the current site without navigating to sites \
like Google. Be forthright when it is impossible to answer the question or carry out the task. \
The assistant will end the task once it sends a message to the user.""",
    'use_nav': False,
    'eval_mode': True,
    'truncate_axtree': False,
    'actor_prompt_type': 'with_memory_concise_instruction',
})

webarena_world_model_config = copy.copy(webarena_config)
webarena_world_model_config.update({
    'planner_type': 'world_model',
    'world_model_prompt_type': 'with_update',
    'planner_search_num_actions': 5,
    'planner_search_depth': 1,
    'planner_critic_num_samples': 20,
})