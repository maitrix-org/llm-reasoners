initial_instruction = """
Please provide a clear and step-by-step solution for a scientific problem in the categories of Chemistry, Physics, or Mathematics. The problem will specify the unit of measurement, which should not be included in the answer. Express the final answer as a decimal number with three digits after the decimal point. Conclude the answer by stating "The answer is therefore \\boxed{[ANSWER]}.

For each instance, you need to do three things. Firstly, for "formulae retrieval", you need to identify the formulae explicitly and implicitly entailed in the problem context. Then there is a "reasoning/calculation process" where you are required to reason step by step based on the identified formulae and problem context. Finally, conclude the answer. For each problem, the output format should incorporate the following components in the corresponding format:

**Formulae retrieval: **
[Formula 1] (the formula required to solve the problem)
[Formula 2] (the second formula required to solve the problem, if any)
...
[Formula n] (the n-th formula required to solve the problem, if any)

**Reasoning/calculation process:**
[step 1] (the first step for solving this problem)
.....
[step n] (the n-th step for solving the problem, if any)

**Answer conclusion:**
[answer] The answer is therefore \\boxed{[ANSWER]}.
"""

refine_formulae_prompt = """

You are provided with a ###chemistry problem### and a ###Formula retrieval### process for solving the problem. 

For each instance, you need to do three things. First, for "judgement of the retrieved formulae", you need to review the provided formulae and give your judgement. Then re-organize the "formulae retrieval" process based on your judgement. Finally, justify your answer with a "confidence score" in the scale of [0,1]. The output format should incorporate these components in the following format:

**Judgement of the retrieved formulae:**
[judgement] (Your assessment of whether the retrieved formulae are correct or not.)

**Formula retrieval:**
(Your revised correct formulae required for the problem.)
[Formula 1] (the formula required to solve the problem)
[Formula 2] (the second formula required to solve the problem, if any)
...
[Formula n] (the n-th formula required to solve the problem, if any)

**Confidence score:**
[score] (float number in [0,1])

"""

refine_reasoning_prompt = """

You are provided with a ###chemistry problem###, the corresponding ###Formula retrieval### and ###Reasoning/calculation process### for solving the problem. 

For each instance, you need to do three things. First, for "judgement of the retrieved formulae", you need to review the provided reasoning process based on the formulae collected and give your judgement. Then re-organize the "reasoning process" based on your judgement. Finally, justify your answer with a "confidence score" in the scale of [0,1]. The output format should incorporate these components in the following format:

**Judgement of the reasoning process:**
[judgement] (Your assessment of whether the reasoning process are correct or not.)

**Reasoning/calculation process:**
(Your revised correct reasoning process to solve the problem based on the given formulae.)
[step 1] (the first step for solving this problem)
.....
[step n] (the n-th step for solving the problem, if any)

**Confidence score:**
[score] (float number in [0,1])

"""

final_answer_prompt = """

Your task is to solve a chemical problem. For each instance, you will be given two sources of information: "formula retrieval" and "reasoning/calculation process". Your task is to conclude the final answer based on these information. The final output template is:

**Answer conclusion:**
[answer] The answer is therefore \\boxed{[ANSWER]}.

Now try to conclude the answer for the following problem:

"""