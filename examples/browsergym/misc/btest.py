import openai
from support_good import get_env, reset_env, step_env, get_clustered_action_proposals, create_logger, log_obs, log_reward, get_browser_action_set, get_parsed_evaluations_of_action_proposals


task_name = "webarena.310"

logger = create_logger(task_name)

seed = 16
action_set = get_browser_action_set(); action_history = []
env = get_env(task_name, action_set, seed)

openai_client = openai.OpenAI(api_key="[api_key]")

obs, env_info = reset_env(env, seed, logger)
reward, terminated, truncated = None, False, False
while not terminated and not truncated:

    action_proposals = get_clustered_action_proposals(obs, action_set, action_history, openai_client, logger=logger)

    print(action_proposals)

    actions_with_eval = get_parsed_evaluations_of_action_proposals(obs, action_proposals, action_set, action_history, openai_client, logger=logger)

    print(actions_with_eval)

    action_with_best_eval = max(actions_with_eval, key=lambda x: x[1])[0]
    action_history.append(action_with_best_eval)

    print("SANITY CHECK")
    obs, reward, terminated, truncated, step_info = step_env(env, action_with_best_eval, logger)

    print("SANITY CHECK END")

    print(reward, terminated, truncated)

print(terminated, truncated, reward)

import time
time.sleep(10)

# env.close()

if reward == 1.0:
    print("TASK COMPLETED SUCCESSFULLY")
else:
    print("TASK FAILED")

# run_task("miniwob.click-test", action_set, seed, openai_client)
# success = run_task("miniwob.login-user", action_set, seed, openai_client)
# success = run_task("miniwob.read-table", action_set, seed, openai_client)

# times out here. not that surprising. 
# success = run_task("miniwob.choose-date", action_set, seed, openai_client)
# print(success)