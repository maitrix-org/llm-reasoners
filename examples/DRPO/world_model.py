import io
from typing import NamedTuple, TypedDict
from collections import defaultdict
from reasoners import WorldModel, LanguageModel as Model, Example
from prompt import eval_prompt, in_context_examples, propose_prompt, eval_format_dict, aspects
import json
import logging
from log_format import prompt_log, output_log, eval_log, metrics_log
from utils import parse_json_output
from sentence_transformers import SentenceTransformer, util


class SubResult(NamedTuple):
    system_prompt: str
    in_context_examples: str
    query: str
    output: str
    eval_dict: dict

PromptAlignState = list[SubResult]
PromptAlignAction = (bool,str)
PromptAlignExample = list

def get_top_k_queries(query, k, embedding_model, icl_query_embeddings):
    query_embed = embedding_model.encode(query)
    icl_sims = []

    icl_queries = list(icl_query_embeddings.keys())

    for i in range(len(icl_queries)):
        icl_query = icl_queries[i]
        icl_sims.append((util.cos_sim(icl_query_embeddings[icl_query], query_embed), icl_query))

    icl_sims.sort(reverse=True)

    return icl_sims[:k]

class PromptAlignWorldModel(WorldModel[PromptAlignState, PromptAlignAction, PromptAlignExample]):
    def __init__(self,
                 base_model: Model,
                 metrics_model: Model,
                 eval_model: Model,
                 initial_system_prompt: str,
                 metrics_cache_path: str,
                 depth: int = 10,
                 ret_icl = True,
                 is_GPT = False,
                 k = 2
    ):
        self.base_model = base_model
        self.metrics_model = metrics_model
        self.eval_model = eval_model
        self.initial_system_prompt = initial_system_prompt
        self.metrics_cache_path = metrics_cache_path
        self.depth = depth
        self.ret_icl = ret_icl
        self.is_GPT = is_GPT
        self.k = k

        if self.ret_icl:
            # Setup for retrieval ICL

            # Loading the embedding model
            self.embedding_model = SentenceTransformer("all-mpnet-base-v2")

            # Loading the examples file
            with open('./data/ICL_optimization/out_16_5.json', 'r') as f:
                self.icl_examples = json.load(f)

            # Query embeddings for similarity search
            self.icl_query_embeddings = {}
            for query in self.icl_examples:
                self.icl_query_embeddings[query] = self.embedding_model.encode(query)

        
        logging.info("PromptAlignWorldModel initialized with depth=%d", depth)

    # separate method to get metrics for a query
    def _get_metrics_for_query(self, query):
        try:
            with open(self.metrics_cache_path, "r") as f:
                metrics_cache = json.load(f)
        except FileNotFoundError:
            metrics_cache = {}
        
        if query in metrics_cache:
            metrics_dict = metrics_cache[query]
            metrics_reason = metrics_dict['aspects_selection']['reasoning']
            metrics = metrics_dict['aspects_selection']['selected_aspects']
        else:
            prompt = propose_prompt.replace("[QUERY]", query)
            logging.debug(prompt_log.format(prompt=prompt)) # Log the prompt
            
            metrics_proposal = self.metrics_model.generate(
                user_prompt=prompt,
                temperature=0, top_p=1, max_new_tokens=2048)
            
            metrics_dict = parse_json_output(metrics_proposal)
            metrics_reason = metrics_dict['aspects_selection']['reasoning']
            metrics = metrics_dict['aspects_selection']['selected_aspects']
            metrics_cache[query] = metrics_dict  # Cache the metrics
            with open(self.metrics_cache_path, "w") as f:
                json.dump(metrics_cache, f, indent=4)
        
        logging.info(metrics_log.format(metrics=json.dumps(metrics_dict, indent=4)))  # Log the metrics
        
        return metrics_reason, metrics

    # separate method to generate model output
    def _generate_model_output(self, query, prompt):

        if not self.ret_icl:
            system_prompt = f"# instructions\n\n{prompt}\n\n{in_context_examples}\n\n"
        else:
            top_icl_queries =  get_top_k_queries(query, self.k, self.embedding_model, self.icl_query_embeddings)
            use_in_context_examples = f"# Query:\n"
            cnt = 0
            for _, icl_query in top_icl_queries:
                use_in_context_examples = use_in_context_examples + icl_query +'\n\n#Answer:\n'
                use_in_context_examples = use_in_context_examples + self.icl_examples[icl_query]

                if cnt < (self.k-1):
                    use_in_context_examples = use_in_context_examples +'\n\n#Query:\n'

                cnt += 1
                system_prompt = f"# instructions\n\n{prompt}\n\n{use_in_context_examples}\n\n"
            
        user_prompt = f"# Query:\n{query}\n\n# Answer:\n<START>"
        full_prompt = system_prompt + user_prompt
        logging.debug(prompt_log.format(prompt=full_prompt))  # Log the prompt
        
        if not self.is_GPT:
            output = self.base_model.generate(
                prompts=full_prompt, 
                temperature=0, top_p=1, max_new_tokens=2048, stop=["<END>", "<END", "<|eot_id|>"]).strip()
        else:
            output = self.base_model.generate(
                system_prompt = system_prompt,
                user_prompt=user_prompt, 
                temperature=0, top_p=1, max_new_tokens=2048, stop=["<END>", "<END", "<|eot_id|>"]).strip()
        
        logging.info(output_log.format(output=output))
        return output

    # separate method to generate eval dict
    def _evaluate_output(self, query, output, metrics_reason, metrics):
        prompt = eval_prompt.replace("[QUERY]", query).replace("[OUTPUT]", output).replace("[ASPECT_REASON]", metrics_reason)
        
        eval_aspects = "\n".join([f"- {k}: {aspects[k]}" for k in metrics])
        eval_format = json.dumps({metric: eval_format_dict[metric] for metric in metrics}, indent=4)
        
        eval_prompt_final = prompt.replace("[ASPECT_LIST]", eval_aspects).replace("[EVAL_DICT]", eval_format)
        
        logging.debug(prompt_log.format(prompt=eval_prompt_final))  # Log the prompt
        
        eval_output = self.eval_model.generate(
            user_prompt=eval_prompt_final, 
            temperature=0, top_p=1, max_new_tokens=2048)
        
        try:
            eval_dict = parse_json_output(eval_output)
        except Exception as e:
            logging.info('Some error occured while parsing.')
            return {}
        
        logging.info(eval_log.format(eval=json.dumps(eval_dict, indent=4)))  # Log the evaluation results
        
        return eval_dict
        
                 
    def init_state(self) -> PromptAlignState:
        # logging
        logging.info("Initializing the state")
        logging.info("The initial system prompt is: %s", self.initial_system_prompt)
        
        # sample a query from the example
        query = self.example[0]
        
        # get the metrics for the query
        metrics_reason, metrics = self._get_metrics_for_query(query)

        # generate the output for base model
        output = self._generate_model_output(query, self.initial_system_prompt)
        
        # evaluate the output
        eval_dict = self._evaluate_output(query, output, metrics_reason, metrics)
        
        return [SubResult(
            system_prompt = self.initial_system_prompt,
            in_context_examples = in_context_examples,
            query = query,
            output = output,
            eval_dict = eval_dict
        )]
        
    
    def step(self, state: PromptAlignState, action: PromptAlignAction) -> PromptAlignState:
        # copy
        state = state.copy()
        
        # sample a query from the example based on the state length
        query = self.example[len(state)]
        
        metrics_reason, metrics = self._get_metrics_for_query(query)
        output = self._generate_model_output(query, action)  # Use action as the new system prompt
        eval_dict = self._evaluate_output(query, output, metrics_reason, metrics)
        
        state.append(SubResult(
            system_prompt = action,
            in_context_examples = in_context_examples,
            query = query,
            output = output,
            eval_dict = eval_dict
        ))
        
        return state, {"eval_dict": eval_dict}
    
    def is_terminal(self, state: PromptAlignState) -> bool:
        # several conditions to check
        # 1. depth
        if len(state) >= self.depth:
            # logging
            logging.info("The state is terminal because it reaches the maximum depth")
            return True
        
        # 2. example is exhausted
        if len(state) >= len(self.example):
            # logging
            logging.info("The state is terminal because the example is exhausted")
            return True
        
        # else, not terminal
        return False    