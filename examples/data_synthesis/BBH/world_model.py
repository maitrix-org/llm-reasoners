from reasoners import WorldModel, LanguageModel, GenerateOutput

OrderQAState_t = list[str]
OrderQAAction_t = str

class OrderQAWorldModel(WorldModel):
    def __init__(self,
                 base_model: LanguageModel,
                 check_endstate_prompt: str,
                 prompt: dict) -> None:
        self.base_model = base_model
        self.check_endstate_prompt = check_endstate_prompt
        self.example = None
        self.prompt = prompt

    def init_state(self) -> OrderQAState_t:
        return []

    def step(self, state: OrderQAState_t, action: OrderQAAction_t) -> tuple[OrderQAState_t, dict]:
        return state + [action], {}

    # FRQ isEnd
    def isEnd(self, state):
        if len(state) == 0:
            return False
        # Construct prompt
        prompt_question = 'Now, the problem statement is:\n' + self.prompt
        reasoning_trace = ''.join(f'\n{state[i]}' for i in range(len(state)))
        prompt_trace = 'The reasoning steps so far are:' + reasoning_trace
        prompt_final = self.check_endstate_prompt + '\n\n' + prompt_question + '\n\n' + prompt_trace + '\n\n' + "Now, identify if the last line of the reasoning gives out the final answer. Your output should ONLY be \"0\" or \"1\". You should first consider the speical case."
        
        # print("isEnd final prompt:" + prompt_final)
        
        output = ""
        cnt = 0
        while output not in ["0", "1"]:
            model_output : GenerateOutput = self.base_model.generate(prompts=[prompt_final], temperature=0)
            output = model_output.text[0]
            cnt+=1
            if cnt > 5:
                return -1
        return int(output)

    def is_terminal(self, state: OrderQAState_t) -> bool:
        if self.isEnd(state):
            print("IS TERMINAL")
            return True
        else:
            print("NOT TERMINAL")
            return False