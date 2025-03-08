# Encoder

encoder_prompt_template = """\
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
    'default': encoder_prompt_template
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

memory_update_prompt_template_dict = {
    'default': memory_update_prompt_template
}

# Policy

policy_prompt_template = """\
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

policy_prompt_template_mini = """\
{memory}

# Current State:
{state}

# Intent:
Describe the action the assistant should take next to carry out the user's \
instruction. \
If the user instruction is successfully carried out, message the user and summarize what has been accomplished. \
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
    'default': policy_prompt_template,
    'mini': policy_prompt_template_mini
}

# World Model

world_model_prompt_template = """\
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

world_model_prompt_template_dict = {
    'default': world_model_prompt_template,
}

# Critic

critic_prompt_template = """\
{memory}

# Final State:
{state}

# Task Success and Progress:
Your task is to evaluate the performance of the agent. Given the agent's instruction, interaction history, the final \
state of the webpage, and the agent's responses to the user if any, your goal is to decide whether the agent's execution \
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

critic_prompt_template_mini = """\
{memory}

# Final State:
{state}

# Task Success and Progress:
Your task is to evaluate the performance of the agent. Given the agent's instruction, interaction history, the final \
state of the webpage, and the agent’s responses to the user if any, your goal is to decide whether the agent \
is successful or not at carrying out the instructions. \
Success includes completion of intermediary tasks as well as fulfillment of the user's instructions \
without significant errors, omissions, or unresolved issues. \
If the current state is a failure but it looks like the agent is on the right track towards \
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

critic_prompt_template_dict = {
    'default': critic_prompt_template,
    'mini': critic_prompt_template_mini
}

actor_prompt_template = """\
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

actor_prompt_template_mini = """\
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
Interact only with elements in the current step observation. \
When filling search boxes, fill into the corresponding combobox instead. Your response \
will be executed as a Python function call, so ensure it adheres to the format \
and argument data type specifications defined in the action space.

Wrap your response in the tag <action> and </action>.\
"""

actor_prompt_template_concise_instruction = """\
{memory}

# Observation:
{observation}

# Current State:
{state}

# Current Intent:
{plan}

# Action:
Choose an API call that will carry out the intent when executed in the webpage. \
Use only one action at a time. You must not enclose bid inputs in [brackets] but instead in 'single quotes'. The bid is a number instead of a word. \
Interact only with elements in the current step observation. \
Don't send the user message if you met errors or missing information or if the \
task can't be completed. Your response \
will be executed as a Python function call, so ensure it adheres to the format \
and argument data type specifications defined in the action space.
If you are sending a message to the user, give very short answer in words, numerics, or the requested url \
and only include the direct answer to the question given in the user instruction.

Wrap your response in the tag <action> and </action>.\
"""

actor_prompt_template_dict = {
    'default': actor_prompt_template,
    'mini': actor_prompt_template_mini,
    'concise_instruction': actor_prompt_template_concise_instruction,
}