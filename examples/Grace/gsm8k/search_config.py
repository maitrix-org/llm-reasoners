import io
import numpy as np

from world_model import GSM8kState, GSM8kAction
from reasoners import SearchConfig, LanguageModel
from typing import Tuple
from tqdm import tqdm
import torch
import re, time
from data_utils.utils import prepare_icl_input
from constants import *
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import top_k_top_p_filtering
from transformers import PreTrainedTokenizer, PreTrainedModel
from torch.nn.utils.rnn import pad_sequence


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
        self.device = "cuda"
        self.discriminator_model = discriminator_model
        self.discriminator_tokenizer = discriminator_tokenizer
        self.example = None
        self.temperature = temperature
        self.n_actions = n_actions
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha

    def get_actions(self, state: GSM8kState) -> list[GSM8kAction]:
        # generate n_actions, here n_actions should be
        print(f"curr state: {state}")

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
                                            eos_token_id="|", #[1820]-> |
                                            num_return_sequences=self.n_actions,
                                            )

         # Extract log probabilities from the outputs and convert to a PyTorch tensor
        log_probs = outputs.log_prob
        disc_probs = torch.tensor([self.calculate_discriminator_score([qn_with_input, model_prefix, outputs.text[i]]) for i in range(self.n_actions)])
        # Apply softmax to convert log probabilities into action probabilities
        action_probs = torch.softmax(log_probs, dim=0).tolist()
        action_disc_probs = torch.softmax(disc_probs, dim=0).tolist()

        return_actions = []
        for i in range(self.n_actions):
            action = outputs.text[i]
            # Check if the action is already in return_actions
            if action in [a[0] for a in return_actions]:
                continue

            # Use the softmax-transformed probabilities
            prob = action_probs[i]
            disc_prob = action_disc_probs[i]
            return_actions.append((action, prob, disc_prob))

        for entry in return_actions:
            print(f"action: {entry[0]}, lm prob: {entry[1]}, disc prob: {entry[2]}")
        return return_actions

    def fast_reward(self, state: GSM8kState, action: GSM8kAction) -> tuple[float, dict]:
        return 0.1*action[2] + 0.9*action[1], {"disc_score":action[2], "lm_score": action[1] }

    def reward(self, state: GSM8kState, 
               action: GSM8kAction,
               action_confidence: float = None,
               **kwargs) -> float:
        
        return 0.1*action[2] + 0.9*action[1], {"disc_score":action[2], "lm_score": action[1] }
 
    

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
            seq_input =  input_text[2]

            seq = self.base_model.tokenizer.encode(seq_input, return_tensors="pt").to(self.device)
            question_ids = self.discriminator_tokenizer.encode(input_text[0], add_special_tokens=False)
            if self.base_model.tokenizer.__class__.__name__ != self.discriminator_model.__class__.__name__:
                prefix_ids_disc = self.base_model.tokenizer.decode( [] if len(prefix_ids_disc)==0 else prefix_ids_disc[0], skip_special_tokens=True)
                prefix_ids_disc = self.discriminator_tokenizer.encode(prefix_ids_disc, add_special_tokens=False)

                seq = self.base_model.tokenizer.decode(seq[0], skip_special_tokens=True)
                seq = self.discriminator_tokenizer.encode(seq, add_special_tokens=False)

            disc_input_ids.append([self.discriminator_tokenizer.cls_token_id] + question_ids + prefix_ids_disc + [self.discriminator_tokenizer.sep_token_id] + seq)
            ## pad the sequences
            disc_input_ids = pad_sequence([torch.tensor(t) for t in disc_input_ids], batch_first=True, padding_value=self.discriminator_tokenizer.pad_token_id).to(self.device) # batch x seq
            disc_attention_mask = disc_input_ids != self.discriminator_tokenizer.pad_token_id # batch x seq
            ## feed to discriminator to obtain scores
            disc_scores = self.discriminator_model.forward_scores(input_ids=disc_input_ids, attention_mask=disc_attention_mask).view(-1)

        return disc_scores