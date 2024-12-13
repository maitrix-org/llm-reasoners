# Encoder

encoder_prompt_template_with_memory = """\
{memory}

# Observation:
{observation}

# State:
Summarize the current state of the webpage observation, focusing on the most \
recent action you took and any errors encountered. Note any dialogs, progress \
indicators, or significant changes such as items in your cart or sites visited. \
Describe the impact of your previous action on the webpage, including any new \
interactive elements. Include any inferred information that may help achieve \
the goal. Information from steps earlier are for reference only. Focus on \
objective description of the current observation and any inferences you can \
draw from it. Report any error messages displayed. Do not include your next \
planned actions; focus solely on providing an objective summary.

Wrap your response in the tag <state> and </state>.\
"""

encoder_prompt_template_no_memory = """\
# Observation:
{observation}

# State:
Describe all the elements in the current webpage observation. Note any dialogs, \
progress indicators, or error messages. Include any interactive elements and their \
values or if they are blank. \
Note any detailed information such as facts, entities, or data that are relevant \
to the task. Report any error messages like whether the last action was correct. \
Try to be as comprehensive and detailed as possible.

Wrap your response in the tag <state> and </state>.\
"""

encoder_prompt_template_dict = {
    'with_memory': encoder_prompt_template_with_memory,
    'no_memory': encoder_prompt_template_no_memory
}

memory_update_prompt_template = """\
{memory}

# State:
{state}

# Action Intent:
{plan}

# Memory Update:
Summarize the changes in the webpage observation that should be remembered for \
achieving your goal and for predicting the next state. Note any new elements, \
any elements no longer visible, or any changes in the content of existing elements. \
Also note if there is no change. Include any other inferred information that may help \
you decide the next action, such as whether an action intent is successful, or whether \
progress has been made or reversed. Do not include your next planned actions. Revise \
your belief from previous history if the current state contradicts it.

Wrap your response in the tag <memory_update> and </memory_update>.\
"""

memory_update_prompt_template_llama = """\
{memory}

# State:
{state}

# Action Intent:
{plan}

# Memory Update:
Concisely summarize the changes in the webpage observation that should be remembered for \
achieving your goal and for predicting the next state. Note any new elements, \
any elements no longer visible, or any changes in the content of existing elements. \
Also note if there is no change. Avoid including irrelevant information. \
Include any other inferred information that may help \
you decide the next action, such as whether the previous action intent is successful, or whether \
progress has been made or reversed. Do not include your next planned actions. Revise \
your belief from previous history if the current state contradicts it.

Wrap your response in the tag <memory_update> and </memory_update>.\
"""

memory_update_prompt_template_dict = {
    'default': memory_update_prompt_template,
    'llama': memory_update_prompt_template_llama
}

# Memory

memory_prompt_template = """\
{memory}

# State:
{state}

# Action Intent:
{plan}

# Updated Memory
Edit your memory to include the key information and reasoning from the current state that should be remembered \
for achieving your goal and for predicting the next state. 

Wrap your response in the tag <updated_memory> and </update_memory>.\
"""

# Policy

policy_prompt_template_no_memory_update = """\
{memory}

# Current State:
{state}

# Intent:
Describe the action the assistant should take next to carry out the user's \
instruction. \
Avoid using phrases such as "To accomplish the goal," "I will," "To \
proceed.". Avoid ending with phrases like "to execute the search." \
Describe one action at a time and avoid combining multiple steps. \
Refrain from mentioning specific element IDs as they may change \
during execution. Limit your response to one phrase and include any details \
that help select the correct action. Be creative and propose novel \
methods to achieve the goal. Avoid creating accounts without user \
permission or providing personal information. Concrete example \
would be "Go to the home page of Google Flights." and "Click on the 'Search' button."

Wrap your response in the following format:

<think>
Your thoughts and reasoning process
</think>

<intent>
Description of the action to perform next
</intent>\
"""

policy_prompt_template_with_memory_update = """\
{memory}

# Current State:
{state}

# Memory Update:
{memory_update}

# Intent:
Describe the action the assistant should take next to carry out the user's \
instruction. \
Avoid using phrases such as "To accomplish the goal," "I will," "To \
proceed.". Avoid ending with phrases like "to execute the search." \
Describe one action at a time and avoid combining multiple steps. \
Refrain from mentioning specific element IDs as they may change \
during execution. Limit your response to one phrase and include any details \
that help select the correct action. Be creative and propose novel \
methods to achieve the goal. Avoid creating accounts without user \
permission or providing personal information. Concrete example \
would be "Go to the home page of Google Flights." and "Click on the 'Search' button."

Wrap your response in the following format:

<think>
Your thoughts and reasoning process
</think>

<intent>
Description of the action to perform next
</intent>\
"""

policy_prompt_template_dict = {
    'no_update': policy_prompt_template_no_memory_update,
    'with_update': policy_prompt_template_with_memory_update
}

# World Model

world_model_prompt_template_no_update = """\
{memory}

# Current State:
{state}

# Current Intent:
{plan}

# Next State:
Your task is to predict the effect of an action by the agent on a webpage. You are given the interaction history, \
the current state of the webpage, and the agent's current intent for what action to take next. The interaction \
history includes the sequence of actions intended by the agent and the resulting changes to the webpage. \
Note that the action intent may not be executed successfully, so you will have to infer whether the action was successful. \
You are required to predict the new changes that will occur on the webpage \
after the agent attempts to execute their intent, such as the appearance of new elements, the disappearance of existing \
elements, or changes in the content of existing elements. The operation type and the element to operate \
will be provided in the prompt.

Wrap your response in the following format:

<next_state>
Based on the interaction history, current state, and current intent, please predict the changes after \
the agent attempts to carry out the intent. Use present tense. Try to be as comprehensive and detailed as possible. \
Avoid starting phrases like "Based on the interaction history, current state, and current intent".
</next_state>\
"""

world_model_prompt_template_with_update = """\
{memory}

# Current State:
{state}

# Memory Update:
{memory_update}

# Action Intent:
{plan}

# Next State:
Describe all the elements in the webpage after the agent attempts to carry out the intent. \
Note that the execution may not be successful, so you will have to infer the result of the action. \
Note any dialogs, progress indicators, or error messages. Include any interactive elements and their \
values or if they are blank. Note any detailed information such as facts, entities, or data that are relevant \
to the task. Report any error messages displayed. Try to be as comprehensive and detailed as possible.

Wrap your response in the following format:

<next_state>
Follow the format of the current state description. Use present tense. \
Avoid starting phrases like "Based on the interaction history, current state, and current intent".
</next_state>\
"""

world_model_prompt_template_no_memory_with_update = """\
# Current State:
{state}

# Memory Update:
{memory_update}

# Action Intent:
{plan}

# Next State:
Describe all the elements in the webpage after the agent attempts to carry out the intent. \
Note that the execution may not be successful, so you will have to infer the result of the action. \
Note any dialogs, progress indicators, or error messages. Include any interactive elements and their \
values or if they are blank. Note any detailed information such as facts, entities, or data that are relevant \
to the task. Report any error messages displayed. Try to be as comprehensive and detailed as possible.

Wrap your response in the following format:

<next_state>
Follow the format of the current state description. Use present tense. \
Avoid starting phrases like "Based on the interaction history, current state, and current intent".
</next_state>\
"""

world_model_prompt_template_no_memory_with_update_with_knowledge = """\
{knowledge}

# Current State:
{state}

# Memory Update:
{memory_update}

# Action Intent:
{plan}

# Next State:
Describe all the elements in the webpage after the agent attempts to carry out the intent. \
Note that the execution may not be successful, so you will have to infer the result of the action. \
Note any dialogs, progress indicators, or error messages. Include any interactive elements and their \
values or if they are blank. Note any detailed information such as facts, entities, or data that are relevant \
to the task. Report any error messages displayed. Try to be as comprehensive and detailed as possible.

Wrap your response in the following format:

<next_state>
Follow the format of the current state description. Use present tense. \
Avoid starting phrases like "Based on the interaction history, current state, and current intent".
</next_state>\
"""

world_model_prompt_template_dict = {
    'no_update': world_model_prompt_template_no_update,
    'with_update': world_model_prompt_template_with_update,
    'no_memory_with_update': world_model_prompt_template_no_memory_with_update,
    'no_memory_with_update_with_knowledge': world_model_prompt_template_no_memory_with_update_with_knowledge
}

# Critic

critic_prompt_template = """\
{memory}

# Final State:
{state}

# Task Success and Progress:
Your task is to evaluate the performance of the agent. Given the agent's instruction, interaction history, the final \
state of the webpage, and the agent’s responses to the user if any, your goal is to decide whether the agent’s execution \
is successful or not. If the current state is a failure but it looks like the agent is on the right track towards \
success, you should also output as such.

Wrap your response in the following format:

<think>
Your thoughts and reasoning process
</think>

<status>
"success" or "failure"
</status>

<on_the_right_track>
"yes" or "no"
</on_the_right_track>\
"""

actor_prompt_template_with_memory = """\
{memory}

# Observation:
{observation}

# Current State:
{state}

# Current Intent:
{plan}

# Action:
Choose an API call that will carry out the intent when executed in the webpage. \
Use only one action at a time. You must not enclose bid inputs in [brackets] but instead in 'single quotes'. \
Interact only with elements in the current step observation. Your response \
will be executed as a Python function call, so ensure it adheres to the format \
and argument data type specifications defined in the action space.

Wrap your response in the tag <action> and </action>.\
"""

actor_prompt_template_with_memory_concise_instruction = """\
{memory}

# Observation:
{observation}

# Current State:
{state}

# Current Intent:
{plan}

# Action:
Choose an API call that will carry out the intent when executed in the webpage. \
Use only one action at a time. You must not enclose bid inputs in [brackets] but instead in 'single quotes'. \
Interact only with elements in the current step observation. Your response \
will be executed as a Python function call, so ensure it adheres to the format \
and argument data type specifications defined in the action space.
If you are sending a message to the user, give very short answer in words, numerics, or the requested url \
and only include the direct answer to the question given in the user instruction.

Wrap your response in the tag <action> and </action>.\
"""

actor_prompt_template_no_memory = """\
# Observation:
{observation}

# Current State:
{state}

# Current Intent:
{plan}

# Action:
Choose an API call that will carry out the intent when executed in the webpage. \
Use only one action at a time. You must not enclose bid inputs in [brackets] but instead in 'single quotes'. \
Interact only with elements in the current step observation. Your response \
will be executed as a Python function call, so ensure it adheres to the format \
and argument data type specifications defined in the action space.

Wrap your response in the tag <action> and </action>.\
"""

actor_prompt_template_with_memory_with_update = """\
{memory}

# Observation:
{observation}

# Current State:
{state}

# Memory Update:
{memory_update}

# Current Intent:
{plan}

# Action:
Choose an API call that will carry out the intent when executed in the webpage. \
Use only one action at a time. You must not enclose bid inputs in [brackets] but instead in 'single quotes'. \
Interact only with elements in the current step observation. Your response \
will be executed as a Python function call, so ensure it adheres to the format \
and argument data type specifications defined in the action space.

Wrap your response in the tag <action> and </action>.\
"""

actor_prompt_template_dict = {
    'with_memory': actor_prompt_template_with_memory,
    'with_memory_concise_instruction': actor_prompt_template_with_memory,
    'no_memory': actor_prompt_template_no_memory,
    'with_memory_with_update': actor_prompt_template_with_memory_with_update
}