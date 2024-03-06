import io
import numpy as np
from prompt import code_prompt
from world_model import GSM8kState, GSM8kAction
from reasoners import SearchConfig, LanguageModel
from typing import Tuple


class GSM8kConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                discriminator_tokenizer: PreTrainedTokenizer,
                discriminator_model: PreTrainedModel,
                 n_actions=20,
                 temperature=0.8,
                 reward_alpha=0.5,
                 depth_limit=6,
                 force_terminating_on_depth_limit=True) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.temperature = temperature
        self.n_actions = n_actions
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha

    def get_actions(self, state: GSM8kState) -> list[GSM8kAction]:
        # generate n_actions, here n_actions should be
        print(f"state: {self.example}")
        print(f"curr state: {state}")
        print()

        model_input = self.example
        model_prefix= ""
        if len(state)>0:
            for each_element in state:
                model_prefix =  model_prefix + " " + each_element.action
        qn_with_input = prepare_icl_input(model_input, demos=None, instruction=None)
        print(f"qn_with_input: {qn_with_input}")
        outputs = self.base_model.generate(inputs=[qn_with_input, model_prefix],
                                            do_sample=True,
                                            temperature=0.8,
                                            top_p=0.95,
                                            max_new_tokens=60,
                                            bad_words_ids= [[0]],
                                            eos_token_id=[1820],
                                            num_return_sequences=self.n_actions,
                                            output_log_probs=True
                                            )
        
        return_actions = []
        for i in range(self.n_actions):
            action = outputs.text[i]
            # de-duplicate if the action is already in return_actions
            if action in [a[0] for a in return_actions]:
                continue

            log_prob = outputs.log_prob[i]
            return_actions.append((action, log_prob))

        return return_actions

    def fast_reward(self, state: GSM8kState, action: GSM8kAction) -> tuple[float, dict]:
        model_input = self.example
        model_prefix= ""
        if len(state)>0:
            for each_element in state:
                model_prefix =  model_prefix  + each_element.action
        qn_with_input = prepare_icl_input(model_input, demos=None, instruction=None)
        disc_score = self.calculate_discriminator_score([qn_with_input, model_prefix, action[0]])
        return 0.1*disc_score + 0.9*action[1], {"disc_score":disc_score, "lm_score": action[1] }

    def reward(self, state: GSM8kState, 
               action: GSM8kAction,
               action_confidence: float = None,
               **kwargs) -> float:
        
        
        model_input = self.example
        model_prefix= ""
        if len(state)>0:
            for each_element in state:
                model_prefix =  model_prefix  + each_element.action
        qn_with_input = prepare_icl_input(model_input, demos=None, instruction=None)
        disc_score = self.calculate_discriminator_score([qn_with_input, model_prefix, action[0]])
        return 0.1*disc_score + 0.9*action[1], {"disc_score":disc_score, "lm_score": action[1] }
 
    

    def calculate_discriminator_score(self, input_text: list[str]) -> torch.Tensor:
        """
        Calculate the discriminator score for a given string.

        Args:
        - input_text (str): The input text string for which to calculate the discriminator score.
        - discriminator (PreTrainedModel): The discriminator model.
        - discriminator_tokenizer (PreTrainedTokenizer): The tokenizer for the discriminator model.
        - device (torch.device): The device on which to perform the calculation.

        Returns:
        - torch.Tensor: The discriminator score for the input text.
        """
        with torch.no_grad():
            disc_input_ids = []
            prefix_ids_disc = [] if input_text[1]=="" else self.base_model.tokenizer.encode(input_text[1], return_tensors="pt").to(self.device)
            # print(f"prefix_ids_disc: {prefix_ids_disc}")
            seq_input =  input_text[2]
            # print(f"seq_input : {seq_input} ")

            seq = self.base_model.tokenizer.encode(seq_input, return_tensors="pt").to(self.device)
            question_ids = self.discriminator_tokenizer.encode(input_text[0], add_special_tokens=False)
            if self.base_model.tokenizer.__class__.__name__ != self.discriminator_model.__class__.__name__:
                prefix_ids_disc = self.base_model.tokenizer.decode( [] if len(prefix_ids_disc)==0 else prefix_ids_disc[0], skip_special_tokens=True)
                prefix_ids_disc = self.discriminator_tokenizer.encode(prefix_ids_disc, add_special_tokens=False)

                # print(f"seq: {seq}")
                seq = self.base_model.tokenizer.decode(seq[0], skip_special_tokens=True)
                seq = self.discriminator_tokenizer.encode(seq, add_special_tokens=False)

            print(f" input_test: {input_text}")
            print(f"question_ids: {question_ids}")
            print(f"prefix_ids_disc: {prefix_ids_disc}")
            print(f" [self.discriminator_tokenizer.sep_token_id]: {[self.discriminator_tokenizer.sep_token_id]}")
            print(f"seq {seq}")
            disc_input_ids.append([self.discriminator_tokenizer.cls_token_id] + question_ids + prefix_ids_disc + [self.discriminator_tokenizer.sep_token_id] + seq)
            ## pad the sequences
            disc_input_ids = pad_sequence([torch.tensor(t) for t in disc_input_ids], batch_first=True, padding_value=self.discriminator_tokenizer.pad_token_id).to(self.device) # batch x seq
            disc_attention_mask = disc_input_ids != self.discriminator_tokenizer.pad_token_id # batch x seq
            ## feed to discriminator to obtain scores
            disc_scores = self.discriminator_model.forward_scores(input_ids=disc_input_ids, attention_mask=disc_attention_mask).view(-1)
            print(f"disc_scores: {disc_scores.shape}")

        return disc_scores
