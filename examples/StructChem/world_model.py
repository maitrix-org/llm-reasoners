import io
from typing import NamedTuple, TypedDict
from collections import defaultdict
from prompt import refine_formulae_prompt, refine_reasoning_prompt
from reasoners import WorldModel, LanguageModel
import os
import re

local_rank = int(os.environ.get("LOCAL_RANK", 0))


class SubResult(NamedTuple):
    formulae: str
    confidence: float

class SubResultReasoning(NamedTuple):
    reasoning: str
    confidence: float


StructChemState = list[SubResult]
StructChemStateReasoning = list[SubResultReasoning]
StructChemAction = list
StructChemExample = str


class StructChemWorldModelF(WorldModel[StructChemState, StructChemAction, StructChemExample]):
    """
    StructChem world model
    State: [[formulae_1, confidence_1], [formulae_n, confidence_n], ...]
    Action: formulae
    """

    def __init__(self,
                 base_model: LanguageModel,
                 temperature=0.7) -> None:
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature

    def init_state(self) -> list:
        return []

    def step(self, state: StructChemState, action: StructChemAction) -> tuple[StructChemState, dict]:
        state = state.copy()

        if len(state) == 0:
            state.append(SubResult(action, 0.0))

        # Search for formulae in a "tree"
        with io.StringIO() as f:
            f.write(refine_formulae_prompt.strip() + "\n\n")
            f.write("Now try the following. Remember to strictly follow the output format:\n\n")
            chemistry_question = self.example.split("<concatenate>")[0]
            f.write(f"### Chemistry problem:###\n {chemistry_question}\n\n### Formulae retrieval:###\n{state[-1].formulae}")
            model_input = f.getvalue()

        refined_formulae = self.base_model.generate(
            [model_input],
            hide_input=True,
            top_k=32000,
            top_p=0.95,
            do_sample=True,
            temperature=self.temperature,
            eos_token_id='\n'
        ).text[0].strip()
        formulae, conf_f = refined_formulae.split("**Confidence score:**")[0].strip("\n"), refined_formulae.split("**Confidence score:**")[1].strip()
        
        # extract the confidence score and the refined components
        conf = float(re.findall(r"\d+\.?\d*", conf_f)[0])
        formulae = "**Formula retrieval:**" + formulae.split("**Formula retrieval:**")[1]

        # local rank
        if local_rank == 0:
            with open("logs/structChem_BeamSearch/log.txt", "a") as f:
                print(f"Formulae: {action[0]}", file=f)
                print(f"Confidence: {action[1]}", file=f)
                print(f"\n", file=f)

        if conf >= state[-1].confidence: 
            state.append(SubResult(formulae, conf))

        return state, {}

    def is_terminal(self, state: StructChemState) -> bool:
        # No special termination conditions except depth of search (number of iterations).
        return False


class StructChemWorldModelR(WorldModel[StructChemStateReasoning, StructChemAction, StructChemExample]):
    """
    StructChem world model
    State: [[reasoning_1, confidence_1], [reasoning_n, confidence_n], ...]
    Action: reasoning
    """

    def __init__(self,
                 base_model: LanguageModel,
                 temperature=0.7) -> None:
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature

    def init_state(self) -> list:
        return []

    def step(self, state: StructChemStateReasoning, action: StructChemAction) -> tuple[StructChemStateReasoning, dict]:
        state = state.copy()

        if len(state) == 0:
            state.append(SubResultReasoning(action, 0.0))

        # Search for formulae in a "tree"
        with io.StringIO() as f:
            f.write(refine_reasoning_prompt.strip() + "\n\n")
            f.write("Now try the following. Remember to strictly follow the output format:\n\n")
            chemistry_question = self.example.split("<concatenate>")[0]
            refined_formulae = self.example.split("<concatenate>")[1]
            f.write(f"### Chemistry problem:###\n {chemistry_question}\n\n### Formulae retrieval:###\n{refined_formulae}\n\n###Reasoning process###\n{state[-1].reasoning}")
            model_input = f.getvalue()

        refined_reasoning = self.base_model.generate(
            [model_input],
            hide_input=True,
            top_k=32000,
            top_p=0.95,
            do_sample=True,
            temperature=self.temperature,
            eos_token_id='\n'
        ).text[0].strip()

        reasoning, conf_f = refined_reasoning.split("**Confidence score:**")[0].strip("\n"), refined_reasoning.split("**Confidence score:**")[1].strip()

        # extract the confidence score and the refined components
        conf = float(re.findall(r"\d+\.?\d*", conf_f)[0])
        reasoning = "**Reasoning/calculation process:**" + reasoning.split("**Reasoning/calculation process:**")[1]

        # local rank
        if local_rank == 0:
            with open("logs/structChem_BeamSearch/log.txt", "a") as f:
                print(f"Reasoning: {action[0]}", file=f)
                print(f"Confidence: {action[1]}", file=f)
                print(f"\n", file=f)

        if conf >= state[-1].confidence: 
            state.append(SubResultReasoning(reasoning, conf))

        return state, {}

    def is_terminal(self, state: StructChemStateReasoning) -> bool:
        # No special termination conditions except depth of search (number of iterations).
        return False
