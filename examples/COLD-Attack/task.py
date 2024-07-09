import pandas as pd
import os

from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))


def load_task_dataset():
    
    data = pd.read_csv("./data/advbench/harmful_behaviors_custom.csv")

    targets = data['target'].tolist()
    goals = data['goal'].tolist()

    xs = []
    zs = []
    for i, d in enumerate(zip(goals, targets)):
        goal = d[0].strip()
        target = d[1].strip()
        
        x = goal.strip()
        z = target.strip()
        
        # _, text, text_post, decoded_text, p_with_adv = decode(model, tokenizer, device, x ,z, None, args, DEFAULT_SYSTEM_PROMPT)
        xs.append(x)
        zs.append(z)
    return xs, zs