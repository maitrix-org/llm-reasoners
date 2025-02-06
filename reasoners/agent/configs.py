import copy

default_web_agent_config = {
    'agent_name': 'Web Browsing Agent',
    'agent_description': """An information and automation assistant who responds to \
user instructions by browsing the internet. The assistant strives to answer each question \
accurately, thoroughly, efficiently, and politely, and to be forthright when it is \
impossible to answer the question or carry out the instruction. The assistant will \
end the task once it sends a message to the user.""",
    'memory_type': 'step_prompted',
    'planner_type': 'policy',
    'policy_output_name': 'intent',
    'module_error_message': 'send_msg_to_user("LLM output parsing error")',
    'with_datetime': True,
    'eval_mode': False,
    'truncate_axtree': True,
    'max_steps': 30
}

browsergym_config = copy.copy(default_web_agent_config)
browsergym_config.update({
    'environment': 'browsergym',
})

browsergym_world_model_config = copy.copy(browsergym_config)
browsergym_world_model_config.update({
    'planner_type': 'world_model',
    'planner_search_num_actions': 5,
    'planner_search_depth': 1,
    'planner_policy_num_samples': 20,
    'planner_critic_num_samples': 20,
})


browsergym_webarena_config = copy.copy(browsergym_config)
browsergym_webarena_config.update({
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
    'actor_prompt_type': 'concise_instruction',
    'with_datetime': False
})

browsergym_webarena_world_model_config = copy.copy(browsergym_webarena_config)
browsergym_webarena_world_model_config.update({
    'planner_type': 'world_model',
    'planner_search_num_actions': 5,
    'planner_search_depth': 1,
    'planner_policy_num_samples': 20,
    'planner_critic_num_samples': 20,
})

easyweb_config = copy.copy(default_web_agent_config)
easyweb_config.update({
    'environment': 'easyweb',
    'use_nav': True,
})

easyweb_world_model_config = copy.copy(easyweb_config)
easyweb_world_model_config.update({
    'planner_type': 'world_model',
    'planner_search_num_actions': 5,
    'planner_search_depth': 1,
    'planner_policy_num_samples': 20,
    'planner_critic_num_samples': 20,
})

easyweb_mini_config = copy.copy(easyweb_config)
easyweb_mini_config.update({
    'agent_description': 'An information and automation assistant who responds to \
user instructions by browsing the internet. The assistant strives to answer each question \
accurately, thoroughly, efficiently, and politely, and to be forthright when it is \
impossible to answer the question or carry out the instruction. The assistant will not attempt to solve CAPTCHAs. \
The assistant will end the task once it sends a message to the user. The assistant remembers that bids \
are numbers in square brackets at the beginning of each line, and that it should click on an option to select it. \
The assistant will focus on providing information for the user and avoid making purchases or bookings. \
After the instruction is successfully carried out, the assistant will message the user to summarize what has been done.',
    'policy_prompt_type': 'mini',
    'critic_prompt_type': 'mini',
    'actor_prompt_type': 'mini',
})

easyweb_mini_world_model_config = copy.copy(easyweb_mini_config)
easyweb_mini_world_model_config.update({
    'planner_type': 'world_model',
    'planner_search_num_actions': 5,
    'planner_search_depth': 1,
    'planner_policy_num_samples': 10,
    'planner_critic_num_samples': 10,
})

easyweb_webarena_config = copy.copy(easyweb_config)
easyweb_webarena_config.update({
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
    'actor_prompt_type': 'concise_instruction',
    'with_datetime': False
})

easyweb_webarena_world_model_config = copy.copy(easyweb_webarena_config)
easyweb_webarena_world_model_config.update({
    'planner_type': 'world_model',
    'planner_search_num_actions': 5,
    'planner_search_depth': 1,
    'planner_policy_num_samples': 20,
    'planner_critic_num_samples': 20,
})

CONFIG_LIBRARY = {
    'browsergym': browsergym_config,
    'browsergym_world_model': browsergym_world_model_config,
    'browsergym_webarena': browsergym_webarena_config,
    'browsergym_webarena_world_model': browsergym_webarena_world_model_config,
    'easyweb': easyweb_config,
    'easyweb_world_model': easyweb_world_model_config,
    'easyweb_mini': easyweb_mini_config,
    'easyweb_mini_world_model': easyweb_mini_world_model_config,
    'easyweb_webarena': easyweb_webarena_config,
    'easyweb_webarena_world_model': easyweb_webarena_world_model_config,
}