import re
from collections import Counter

def extract_subquestions(text):
    # Remove any substrings enclosed in parentheses
    text_without_parentheses = re.sub(r'\(.*?\)', '', text)

    # Look for the phrase "we need to know", and then find all subsequent substrings in quotes
    pattern = r'we need to know:(.*)'
    match = re.search(pattern, text_without_parentheses, re.IGNORECASE)
    if not match:
        return []
    
    # Extract all substrings enclosed in double quotes from the matched part
    subquestions = re.findall(r'\"(.*?)\"', match.group(1))
    return subquestions

def majority_voting(outputs):
    # Return the most common output
    return Counter(outputs).most_common(1)[0][0]

def judge_answer(pred, label):
    # pred is yes or no (string)
    # label is true or false (boolean)
    if pred == 'yes':
        return label
    elif pred == 'no':
        return not label

def parse_answer(text):
    match = re.search(r"the answer is (\w+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None