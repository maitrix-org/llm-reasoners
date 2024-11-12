import os
import sys
import fire
import logging
import json
from typing import Type, Optional

from .icl_opti_prompts import eval_prompt, propose_prompt, optimize_prompt, eval_format_dict, aspects
from .utils import parse_json_output

from reasoner.models import OpenAIChatModel

def get_aspects(query, metrics_model, metrics_cache_path):

    # Reading in metrics cache
    try:
        with open(metrics_cache_path, "r") as f:
            metrics_data_cache = json.load(f)
    except FileNotFoundError:
        metrics_data_cache = {}

    
    # Finding aspects
    if query in metrics_data_cache:
        metrics_reason = metrics_data_cache[query]['aspects_selection']['reasoning']
        metrics = metrics_data_cache[query]['aspects_selection']['selected_aspects']

        return metrics, metrics_reason

    
    else:
        prompt = propose_prompt.replace("[QUERY]", query)

        metrics_proposal = metrics_model.generate(
                    user_prompt=prompt,
                    temperature=0, top_p=1, max_new_tokens=2048)
        
        metrics_dict = parse_json_output(metrics_proposal)
        metrics_reason = metrics_dict['aspects_selection']['reasoning']
        metrics = metrics_dict['aspects_selection']['selected_aspects']
        metrics_data_cache[query] = metrics_dict  # Cache the metrics

        with open(metrics_cache_path, "w") as f:
            json.dump(metrics_data_cache, f, indent=4)

        return metrics, metrics_reason

def evaluate_resp(query, output, metrics, metrics_reason, eval_model):
    prompt = eval_prompt.replace("[QUERY]", query).replace("[OUTPUT]", output).replace("[ASPECT_REASON]", metrics_reason)
        
    eval_aspects = "\n".join([f"- {k}: {aspects[k]}" for k in metrics])
    eval_format = json.dumps({metric: eval_format_dict[metric] for metric in metrics}, indent=4)
    
    eval_prompt_final = prompt.replace("[ASPECT_LIST]", eval_aspects).replace("[EVAL_DICT]", eval_format)

    eval_output = eval_model.generate(
            user_prompt=eval_prompt_final, 
            temperature=0, top_p=1, max_new_tokens=2048)
    
    eval_dict = parse_json_output(eval_output)

    return eval_dict

def optimize_resp(query, output, eval_dict, optimize_model):
    prompt = optimize_prompt.replace("[QUERY]", query)\
                                .replace("[OUTPUT]", output)\
                                .replace("[OUTPUT_EVALUATION]", json.dumps(eval_dict, indent=4))
    
    outputs = optimize_model.generate(
            user_prompt = prompt,
            temperature = 0,
            top_p = 1,
            max_new_tokens = 2048,
            num_return_sequences = 1
        )
    
    outputs = parse_json_output(outputs)
    
    return outputs
                
def get_total_score(eval_dict):
    score = 0
    for aspect in eval_dict.keys():
        score += int(eval_dict[aspect]['score'])

    return score


def run_icl_align(
        eval_model,
        metrics_model,
        optimize_model,
        num_iters,
        log_dir,
        data_dir,
        metrics_cache_path,
        num_samples = 16):
    
    output_path = log_dir + '/out_' + str(num_samples) + '_' + str(num_iters) + '.json'
    final_result_dict = {}
    
    with open(os.path.join(data_dir, 'queries_resp_train.json')) as f:
        icl_train_data = json.load(f)

    queries = list(icl_train_data.keys())

    for i in range(num_samples):
        query = queries[i]
        output = icl_train_data[query]

        logging.info('------------------------------Query---------------------------')
        logging.info(query)

        metrics, metrics_reason = get_aspects(query, metrics_model, metrics_cache_path)
        logging.info('------------------------------Metrics Chosen-------------------')
        logging.info(str(metrics))
        logging.info('------------------------Step-0-----------------------')
        logging.info('----------------Output------------')
        logging.info(output)
        logging.info('----------------Evaluation--------')

        try: 
            eval_dict = evaluate_resp(query, output, metrics, metrics_reason, eval_model)
        except:
            eval_dict = evaluate_resp(query, output, metrics, metrics_reason, eval_model)

        logging.info(json.dumps(eval_dict, indent=4))
        logging.info('-----------------------------------------------------')

        best_score = get_total_score(eval_dict)
        final_result_dict[query] = output

        for i in range(1, num_iters+1):
            logging.info('------------------------Step-{}-----------------------'.format(i))
            logging.info('----------------Optimization--------')

            try:
                optimized_output_dict = optimize_resp(query, output, eval_dict, optimize_model)
            except:
                optimized_output_dict = optimize_resp(query, output, eval_dict, optimize_model)

            output = optimized_output_dict['new_response']
            logging.info(output)
            logging.info('----------------Evaluation--------')

            try:
                eval_dict = evaluate_resp(query, output, metrics, metrics_reason, eval_model)
            except: 
                eval_dict = evaluate_resp(query, output, metrics, metrics_reason, eval_model)

            logging.info(json.dumps(eval_dict, indent=4))
            logging.info('-----------------------------------------------------')
            
            if get_total_score(eval_dict) >= best_score:
                best_score = get_total_score(eval_dict)
                final_result_dict[query] = output

            if best_score == 5*len(eval_dict.keys()):
                logging.info('All scores have been maximized stopping optimization')
                break
        
        with open(output_path, 'w') as f:
            json.dump(final_result_dict, f, indent=4)

    logging.info('The ICL Optimization has completed.')


def main(
    eval_model_name: str = 'gpt-4-0125-preview',
    metrics_model_name: str = 'gpt-4-0125-preview',
    optimize_model_name: str = 'gpt-4-0125-preview',
    num_iters = 5,
    log_dir: Optional[str] = "logs/ICL_optimization",       
    data_dir = './data',
    metrics_cache_path: str = "./data/icl_metrics_cache.json",
    logging_level: str = "INFO",
    num_samples = 16,
):
    # if log_dir is not None, create the directory
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        
    # if metrics_cache_path is not None and it does not exist, create it
    if metrics_cache_path is not None and not os.path.exists(metrics_cache_path):
        with open(metrics_cache_path, "w") as f:
            json.dump({}, f)
        
    # set up logging
    logging_text_file = os.path.join(log_dir, 'log.txt')
    
    # clear it anyway
    with open(logging_text_file, 'w'):
        pass
    
    logging.basicConfig(
        level=logging.INFO if logging_level == "INFO" else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logging_text_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Initialize a dictionary to hold model instances
    models = {}

    # Always create the eval model
    models['eval'] = OpenAIChatModel(model_name=eval_model_name)

    # Reuse the eval model for optimize and metrics models if their names match, otherwise create new instances
    for model_type, model_name in [('optimize', optimize_model_name), ('metrics', metrics_model_name)]:
        if model_name in models.values():
            # Reuse the existing model instance if the name matches
            models[model_type] = models['eval']
        else:
            # Create a new model instance if the name does not match
            models[model_type] = OpenAIChatModel(model_name=model_name)

    # Access models as needed
    eval_model = models['eval']
    optimize_model = models['optimize']
    metrics_model = models['metrics']


    run_icl_align(
        eval_model=eval_model,
        metrics_model=metrics_model,
        optimize_model=optimize_model,
        num_iters=num_iters,
        log_dir=log_dir,
        data_dir=data_dir,
        metrics_cache_path=metrics_cache_path,
        num_samples = num_samples
    )
    
    
if __name__ == '__main__':
    fire.Fire(main)