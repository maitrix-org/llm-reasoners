import argparse
import json
import os
from glob import glob
# from main import main
import pandas as pd

def get_fanoutqa_dataset(data_root):
    data_path = os.path.join(data_root, 'fanout-dev-0-30.json')
    with open(data_path) as f:
        data = json.load(f)
    
    questions = [row['question'] for row in data]
    return questions

def get_flightqa_dataset(data_root):
    data_path = os.path.join(data_root, 'flight_questions_train.jsonl')
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))

    questions = [row['question'] for row in data if row['level'] < 3]
    return questions

def get_flightqa_new_dataset(data_root, filename='flight_search_questions_new.csv'):
    data_path = os.path.join(data_root, filename)
    data_df = pd.read_csv(data_path)
    questions = data_df['question'].tolist()
    return questions

def get_webarena_dataset(data_root):
    data_path = os.path.join(data_root, 'webarena_test.json')
    with open(data_path) as f:
        data = json.load(f)
    return data

def get_dataset(dataset, data_root): 
    if dataset == 'fanout':
        questions = get_fanoutqa_dataset(data_root)
    elif dataset == 'flightqa':
        questions = get_flightqa_dataset(data_root)
    elif dataset == 'webarena':
        questions = get_webarena_dataset(data_root)
    elif dataset == 'flightqa_counterfactual_ablation_30':
        questions = get_flightqa_new_dataset(data_root, filename='flightqa_counterfactual_expansion.csv')
    else: 
        raise ValueError(f'Invalid dataset: {dataset}')
    return questions