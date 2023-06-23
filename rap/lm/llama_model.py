from ..rap import LanguageModel
import time, json, os, sys
from pathlib import Path
from typing import Tuple
import torch
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import numpy as np

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
    return local_rank, world_size


class LLaMAModel(LanguageModel):

    def __init__(self, path, size, max_batch_size=1, max_seq_len=2048, local_rank=-1, world_size=-1):
        super().__init__()
        if local_rank == -1 or world_size == -1:
            local_rank, world_size = setup_model_parallel()
        self.tokenizer, self.model = self.load(os.path.join(path, size), os.path.join(path, "tokenizer.model"), local_rank, world_size, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
    
    #  self.tokenizer, self.model = self.load(os.path.join(path, size), os.path.join(size, "tokenizer.model"), local_rank, world_size, max_batch_size=max_batch_size, max_seq_len=max_seq_len)                                                               
    # TypeError: LLaMAModel.load() got multiple values for argument 'max_batch_size'


    def load(self, ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, max_batch_size: int, max_seq_len: int) -> Tuple[Tokenizer, Transformer]:
        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert (
                world_size == len(checkpoints)
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        ckpt_path = checkpoints[local_rank]
        print("Loading")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args).cuda().half()
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        return tokenizer, model

    @torch.no_grad()
    def __call__(
        self,
        inputs: list[str],
        max_gen_len: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        end_token: str = "",  # TODO: change this to a function
        return_probs: bool = False,
        hide_input: bool = False,
    ) -> dict:
        """Generate text from a list of prompts.
        
        Args:
            inputs (list[str]): List of prompts.
            max_gen_len (int): Maximum length of generated text.
            temperature (float, optional): Temperature for sampling. 0 for greedy decoding. Defaults to 0.8.
            top_p (float, optional): Top-p for sampling. Defaults to 0.5.
            eos_token_id (int, optional): Token id for end of sentence. Defaults to -100.
        """
        if end_token == "":
            eos_token_id = -100
        else:
            eos_token_id = self.tokenizer.encode(end_token, bos=False, eos=False)[-1]
        end_pos = torch.zeros(len(inputs)).long().cuda() - 1
        bsz = len(inputs)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in inputs]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        assert max_prompt_size <= params.max_seq_len, f"Prompts exceed context length limit: {(max_prompt_size, params.max_seq_len)}"
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t)[:params.max_seq_len].long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0

        eos_cnt = torch.zeros(bsz).long().cuda()
        seq_probs = []
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            seq_probs.append(probs[:, next_token].diag())
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            eos_cnt = eos_cnt + (next_token == eos_token_id).long()
            for idx in range(bsz):
                if eos_cnt[idx] > 0 and end_pos[idx].item() == -1:
                    end_pos[idx] = cur_pos
            if (eos_cnt >= 1).all():
                break
        seq_probs = torch.stack(seq_probs, dim=1) 
        decoded = []
        log_prob = torch.log(seq_probs)
        
        mask = torch.zeros_like(log_prob)
        for i, t in enumerate(tokens.tolist()):
            t = t[:params.max_seq_len]
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            t = [x if x != self.tokenizer.pad_id else self.tokenizer.eos_id for x in t]
            if end_pos[i].item() != -1:
                t = t[: end_pos[i]]
            decoded.append(self.tokenizer.decode(t))
        log_prob = log_prob * mask
        if hide_input:
            decoded = [x[len(inputs[i]):] for i, x in enumerate(decoded)]
        return_dict = {"text": decoded}
        if return_probs:
            return_dict["log_prob"] = log_prob
        
        return return_dict

    @torch.no_grad()
    def get_ll(
        self,
        prefix: str,
        contents: list[str],
    ) -> np.ndarray:
        """Get the log likelihood of the contents given the prefix.
        
        Args:
            prefix (str): The prefix to be excluded from the log likelihood.
            contents (list[str]): The contents to evaluate (must include the prefix).
        """

        params = self.model.params
        bsz = len(contents)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        prefix_tokens = self.tokenizer.encode(prefix, bos=True, eos=False)
        prompts_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in contents]
        # print("prefix length:", len(prefix_tokens))
        for prompt_tokens in prompts_tokens:
            # print("prompt length:", len(prompt_tokens))
            assert prompt_tokens[: len(prefix_tokens)] == prefix_tokens

        max_prompt_size = max([len(t) for t in prompts_tokens])
        total_len = max_prompt_size
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts_tokens):
            tokens[k, : len(t)] = torch.tensor(t)[:params.max_seq_len].long()

        _, h = self.model.forward(tokens[:, :], 0)
        logits = self.model.output(h)
        acc_probs = torch.zeros(bsz).cuda()
        for i in range(len(prefix_tokens), max_prompt_size):
            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != self.tokenizer.pad_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])
        
        return acc_probs.cpu().numpy()


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
