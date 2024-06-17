from datasets import load_dataset
import random
import re
#Function to load the dataset
#Need return three sub datasets for training,evaluating and testing.
#para:None
def load_task_dataset():
    dataset_name="bigbio/med_qa"
    dataset = load_dataset(dataset_name,trust_remote_code=True)
    new_dataset = dict(train=[], test=[])

    def process_split(split_name):
        for example in dataset[split_name]:
            # Extract choices and answer key from the example
            choices = [option['value'] for option in example['options']]
            answer_dict = {option['value']: option['key'] for option in example['options']}
            
            # Construct the question format with letters in front of options
            options_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
            question_format = "{question}\nOptions:\n" + options_str
            question_str = question_format.format(question=example['question'])
            
            # Append to the new dataset
            new_dataset[split_name].append(dict(question=question_str, answer=answer_dict[example['answer']]))

    process_split('train')
    process_split('test')
    dataset = new_dataset
    random.seed(30)
    random.shuffle(dataset['train'])
    random.shuffle(dataset['test'])
    questions_train = dataset['train'][:2000]
    questions_eval = dataset['train'][2000:2150]
    questions_test= dataset['test'][0:500]
    return questions_train,questions_eval,questions_test

#reformat the dataset before passing to the prompt agent
#para: question_list
def reformat_data(question_list):
    return question_list

#function to extract the answer from the response by LLM
def extract_answer(message):
    pattern = r"<answer>\s*([A-Za-z])\s*\..*?</answer>"
    answer = re.search(pattern, message)
    if answer == None:
        pattern = r"<answer>\s*([A-Za-z])\s*</answer>"
        answer = re.search(pattern, message)
    if answer == None:
        pattern = r"<([A-Za-z])>"
        answer = re.search(pattern, message)
    if answer:
        answer = answer.group(1)
    return answer

#function to check the answer correct or not
def check_anwser(model_answer,correct_answer):
    if model_answer==correct_answer:
        return True
    return False
