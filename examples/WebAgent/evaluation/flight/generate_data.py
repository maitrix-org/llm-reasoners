from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
import json
import sys
import pandas as pd
import os
import sys

# Get the directory of the current file
current_directory = os.path.dirname(os.path.abspath(__file__))
default_api_key_path = os.path.join(current_directory, '..', '..', 'default_api_key.txt')
if os.path.exists(default_api_key_path):
    DEFAULT_API_KEY = open(default_api_key_path).read().strip()
else:
    DEFAULT_API_KEY = os.environ.get('OPENAI_API_KEY', None)
    
client = OpenAI(api_key=DEFAULT_API_KEY)

current_datetime = datetime.now().strftime('%a, %b %d, %Y %H:%M:%S')

system_prompt = f'You are a creative writer who is an expert at crafting questions to help train assistants who answer user queries. Current date and time: {current_datetime}'

instruction_prompt = """\
Your task is to create a robust benchmark for evaluating an AI's ability to search for flights through a platform like Google Flights/ \
To ensure the dataset effectively represents real-world use cases. Here are some important factors to consider:

1. Diversity of Queries
- Range of Destinations: Include both common and obscure destinations to test how well the model handles varying levels of demand.
- Dates and Duration: Include different date ranges, including last-minute flights, peak travel dates (like holidays), and extended trips. Ensure there’s a variety in trip duration as well.
- Passenger Variability: Include solo travelers, families, and group travel (e.g., one adult vs. two adults and two children) since these change the search parameters and price results.
- Class and Preference: Vary preferences like cabin class (economy, business, first) and filters (non-stop, one stop, preferred airlines, etc.).
- Budget Constraints: Test price sensitivity by setting different budget limits to see how well the AI handles trade-offs.

2. Complexity of Requirements
- Multi-Leg Flights: Add queries for multi-city trips or those requiring complex layovers.
- Dynamic Constraints: Include queries with dynamic constraints, such as “find the cheapest flight but depart between 8-10 AM,” to see if the model can adapt its search within specific time frames.
- Conditional Preferences: Test cases where users might want options based on multiple conditions, like “either find the cheapest non-stop or the shortest two-stop option.”

In practice, the questions typically vary in the following dimensions: 
- Ticket type (round-trip, one-way, etc.)
- Routes (origin and destination)
- Layover location(s)
- Dates (departure and/or return)
- Flight time (departure and arrival)
- Total flight time
- Airlines
- Cabin class (economy, business, etc.)
- Aircraft
- Eco-friendly options (CO2 Emissions)

Given a number of constraints, \
you should first provide a list of constraints, with the number of constraints equal to the specification. \
After that, you will generate a question a typical user will ask which imposes those constraints. \
You should repeat this for at least 7 times to generate a set of questions with simple language. \
Make sure that the number of constraints in the question matches the number of constraints specified.

Do not include constraints about the number of passengers. \
If the constraint is a date, you can use relative dates (e.g., "tomorrow", "next month", "after 8 PM", etc.). \
Avoid using specific dates like "December 25th" to ensure the questions are relevant throughout the year.

Your response should follow the JSON format below: 

Number of Constraints: <num_constraints>

{
    "num_constraints": <num_constraints>,
    "questions": [
        {
            "constraints": [<constraints>], 
            "question": <question>
        },
        ...
    ]
}

Below is a concrete example:

Number of Constraints: 3
    
{
    "num_constraints": 3,
    "questions": [
        {
            "constraints": ["one-way", "New York to London", "departing next Friday"], 
            "question": "Can you find a one-way flight from New York to London departing next Friday?"
        },
        ...
    ]
}\
"""

input_template = """\
Your Response:

Number of Constraints: {num_constraints}
"""

def get_questions(num_constraints, num_questions): 
    questions = []
    while len(questions) < num_questions:
        prompt = instruction_prompt + "\n\n" + input_template.format(num_constraints=num_constraints)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"}
            )
            response = completion.choices[0].message.content
            
            data = json.loads(response)
            
            questions += data['questions']
        except Exception as e:
            print(e)
            continue
    return questions

def collect_data(all_num_constraints, num_questions): 
    num_constraints_to_questions = {}
    for num_constraints in tqdm(all_num_constraints):
        questions = get_questions(num_constraints, num_questions)
        num_constraints_to_questions[num_constraints] = questions
        
    # Filter for banned months
    banned_months = ['January', 'February', 'March', 'April', 
                     'May', 'June', 'July', 'August', 
                     'September', 'October', 'November', 'December']
    banned_months = [month.lower() for month in banned_months]
    
    data = []
    for num_constraints, questions in num_constraints_to_questions.items():
        for question in questions[:]:
            contains_banned_months = any([question['question'].lower().split().count(month) > 0 \
                                        for month in banned_months])
            if contains_banned_months:
                continue
            data.append(question)
    return data

# data = collect_data([3, 4, 5, 6])
data = collect_data([3], 20)
data_df = pd.DataFrame(data)
data_df['num_constraints'] = data_df['constraints'].apply(len)

print(data_df.num_constraints.value_counts())

for i, row in data_df.iterrows():
    print(f"Question {i + 1}: {row['question']}")
    print(f"Constraints: {row['constraints']}")
    print()
    
current_directory = os.path.dirname(os.path.abspath(__file__))

data_df.to_csv(os.path.join(current_directory, '..', '..', 'task_data', 'flightqa_3constraint.csv'), index=False)

# data_df_20 = data_df.groupby('num_constraints').apply(lambda x: x.head()).reset_index(drop=True)
# data_df_20.to_csv('../task_data/flight_search_questions_no_pass_rel_date_20.csv', index=False)