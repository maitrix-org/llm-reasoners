from prompts.game24 import *
import backoff
import openai
import sys
import re
import random
def error_log(details):
    print(f"错误：{details['exception']}", file=sys.stderr)

@backoff.on_exception(backoff.expo, openai.OpenAIError,on_backoff=error_log)
def completions_with_backoff(**kwargs):
    # print("Completion")
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    # print("QWQ")
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    # print("QAQ")
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        try:
            res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        except openai.error.OpenAIError as e:
            print(f"最终失败：{e}", file=sys.stderr)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
    return outputs

def cot_prompt_wrap(x: str, y:str='') -> str:
    return cot_prompt.format(input=x) + y
    
def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]
    
def propose_prompt_wrap(x: str, y: str='') -> str:
    current_numbers = get_current_numbers(y if y else x)
    if current_numbers == '24':
        prompt = cot_prompt.format(input=x) + 'Steps:' + y
    else:
        prompt = propose_prompt.format(input=current_numbers)
    return prompt



def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value
    

    
def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]
    
def value_prompt_wrap(x: str, y: str) -> str:
    last_line = y
    
    if 'left: ' not in last_line:  # last step
        ans = last_line.lower().replace('answer: ', '')
        return value_last_step_prompt.format(input=x, answer=ans)
    current_numbers = get_current_numbers(y)
    return value_prompt.format(input=current_numbers)

def calculate_and_complete_expression(expression_str,numbers_list):
    """
    计算给定字符串中的数学表达式，并返回包含答案的完整表达式。

    参数:
    expression_str (str): 包含数学表达式的字符串，格式为 "1 + 1 = "

    返回:
    str: 计算后包含答案的完整表达式，例如 "1 + 1 = 2"
    """
    try:
        if '=' not in expression_str:
            return False, None, None
        extracted_numbers = re.findall(r'\b\d+\b', expression_str.split('=')[0])
        if len(extracted_numbers) != 2:
            return False, None, None
        num_l = numbers_list[:]
        for num in extracted_numbers:
            if num in num_l:
                num_l.remove(num)
            else: return False, None, None
        left = expression_str.split('=')[0].strip()
        left = left.replace(' ','')
        l = re.split('\+|-|\*|/',left)[0]
        r = re.split('\+|-|\*|/',left)[1]
        op = left[len(l)]
        left = l + ' ' + op + ' ' + r
        math_expression = expression_str.split('=')[0].strip()

        # 使用eval函数计算表达式的结果
        result = eval(math_expression)
        if (result % 1 != 0): return False, None, None
        else: result = int(result)
        num_l.append(str(result))
        lf = ' '.join(num_l)
        complete_expression = left + ' = ' + str(result) + ' (left: ' + lf + ')'

        return True, complete_expression, num_l
    except: return False, None,None
def generate_op(num_list):
    ops = ['+','-','*','/']
    ans = 0.1
    print("Code Generate")
    # print(num_list)
    num_l = num_list[:]
    while(ans < 0 or not isinstance(ans,int)):
        op = random.choice(ops)
        nums = random.sample(num_l,2)
        try:
            ans = eval(nums[0]+op+nums[1])
        except:
            continue
    num_l.remove(nums[0])
    num_l.remove(nums[1])
    num_l.append(str(ans))
    lf = ' '.join(num_l)
    ans_str = nums[0] + ' ' + op + ' ' + nums[1] + ' = ' + str(ans) + ' (left: ' + lf + ')'
    return ans_str, num_l


def can_success(nums):
    def dfs(nums):
        if len(nums) == 1:
            if abs(nums[0] - 24) == 0:
                # print(len(res))
                return True
            else: return False
        for i in range(len(nums)):
            for j in range(len(nums)):
                if j == i:
                    continue
                a, b = nums[i], nums[j]
                for op in '+-*/':
                    if (op == '+' or op == '*') and j > i:
                        continue
                    if op == '/' and a < 1e-6:
                        continue
                    if op == '/' and b % a != 0:
                        continue
                    c = b + a if op == '+' else b - a if op == '-' else b * a if op == '*' else b / a
                    c = int(c)
                    if j > i:
                        next_nums = [c] + nums[:i] + nums[i + 1:j] + nums[j + 1:]
                    else:
                        next_nums = [c] + nums[:j] + nums[j + 1:i] + nums[i + 1:]
                    if(dfs(next_nums)):
                        return True
        return False
    return dfs(nums)