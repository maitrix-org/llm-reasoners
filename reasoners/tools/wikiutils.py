import re

def step(env, action):
    attempts = 0
    while attempts < 1:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1
            
def webthink(env, step_idx, action, thought):
    obs, r, done, info = step(env, action[0].lower() + action[1:])
    step_str = f"Thought {step_idx}: {thought}\nAction {step_idx}: {action}\nObservation {step_idx}: {obs}\n"
    # print(step_str,r,info)
    return step_str[11:]