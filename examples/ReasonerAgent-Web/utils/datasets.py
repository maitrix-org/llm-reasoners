import argparse
import json
import os
from glob import glob
# from main import main
import pandas as pd
import random

def get_fanoutqa_dataset(data_root, 
    filename='fanout-final-dev.json',
    shuffle=False, 
    seed=42, 
    start_idx=0, 
    end_idx=-1,
):
    data_path = os.path.join(data_root, filename)
    with open(data_path) as f:
        data = json.load(f)
    
    questions = [row['question'] for row in data]
    if shuffle:
        random.Random(seed).shuffle(questions)
    return questions[start_idx:end_idx]

def get_flightqa_dataset(data_root, 
    filename='flightqa_counterfactual.csv',
    shuffle=False, 
    seed=42, 
    start_idx=0, 
    end_idx=-1,
):
    data_path = os.path.join(data_root, filename)
    data_df = pd.read_csv(data_path)
    questions = data_df['question'].tolist()
    if shuffle:
        random.Random(seed).shuffle(questions)
    return questions[start_idx:end_idx]

def get_webarena_env_ids(
    shuffle=False, 
    seed=42, 
    start_idx=0, 
    end_idx=-1,
):
    import browsergym.webarena
    env_ids = [
            id for id in gym.envs.registry.keys() if id.startswith('browsergym/webarena')
    ]
    if shuffle:
        random.Random(seed).shuffle(env_ids)
    return sorted(env_ids[start_idx:end_idx], key=lambda s: int(s.split('.')[-1]))

def get_dataset(
    dataset, 
    data_root='', 
    shuffle=False, 
    seed=42, 
    start_idx=0, 
    end_idx=-1,
): 
    if dataset == 'fanout':
        questions = get_fanoutqa_dataset(data_root, shuffle=shuffle, seed=seed, start_idx=start_idx, end_idx=end_idx)
    elif dataset == 'flightqa':
        questions = get_flightqa_dataset(data_root, shuffle=shuffle, seed=seed, start_idx=start_idx, end_idx=end_idx)
    elif dataset == "webarena":
        questions = get_webarena_env_ids(shuffle=shuffle, seed=seed, start_idx=start_idx, end_idx=end_idx)
    else: 
        raise ValueError(f'Invalid dataset: {dataset}')
    return questions