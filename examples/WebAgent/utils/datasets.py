import argparse
import json
import os
from glob import glob
# from main import main
import pandas as pd

def get_fanoutqa_dataset(data_root, filename='fanout-final-dev.json'):
    data_path = os.path.join(data_root, filename)
    with open(data_path) as f:
        data = json.load(f)
    
    questions = [row['question'] for row in data]
    return questions

def get_flightqa_dataset(data_root, filename='flightqa_counterfactual.csv'):
    data_path = os.path.join(data_root, filename)
    data_df = pd.read_csv(data_path)
    questions = data_df['question'].tolist()
    return questions

def get_dataset(dataset, data_root): 
    if dataset == 'fanout':
        questions = get_fanoutqa_dataset(data_root)
    elif dataset == 'flightqa':
        questions = get_flightqa_dataset(data_root)
    else: 
        raise ValueError(f'Invalid dataset: {dataset}')
    return questions