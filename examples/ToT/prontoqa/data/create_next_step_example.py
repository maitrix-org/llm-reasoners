# import json

# def convert_to_next_steps(example_pool):
#     next_steps_list = []

#     for example in example_pool:
#         facts = example['question'].split('. ')[:-1]
#         query = example['query']
#         chain_of_thought = example['chain_of_thought']

#         claims = []
#         next_steps = []

#         for j in range(0, len(chain_of_thought), 2):
#             claim = chain_of_thought[j]
#             next_step = 'Finish.' if j + 1 >= len(chain_of_thought) else chain_of_thought[j + 1]

#             claims.append(claim)
#             next_steps.append(next_step)

#         next_steps_entry = {
#             'Facts': '. '.join(facts) + '.',
#             'Query': query,
#             'claims': claims,
#             'next_steps': next_steps
#         }

#         next_steps_list.append(next_steps_entry)

#     return next_steps_list

# # Read the example pool from a JSON file
# with open('example_pool.json', 'r') as file:
#     example_pool = json.load(file)['example_pool']

# # Process the data
# next_steps = convert_to_next_steps(example_pool)

# # Write the next steps to a JSON file
# with open('example_next_steps.json', 'w') as file:
#     json.dump({"next_steps": next_steps}, file, indent=4)

# print("Next steps have been saved to next_steps.json")


import json
import random

with open(file_path, 'r') as file:
    data = json.load(file)['next_steps']

    # Sample 'sample_size' entries from the data
    sampled_data = random.sample(data, sample_size)


def format_examples(sampled_data):
    formatted_examples = ""
    for i, entry in enumerate(sampled_data, 1):
        facts = f"Facts {i}: {entry['Facts']}\n"
        query = f"Query {i}: {entry['Query']}\n"
        claims_and_next = ""

        for j, (claim, next_step) in enumerate(zip(entry['claims'], entry['next_steps']), 1):
            claims_and_next += f"Claim {i}.{j}: {claim}\nNext {i}.{j}: {next_step}\n"

        formatted_examples += facts + query + claims_and_next + "\n"

    return formatted_examples

# Example usage
file_path = 'example_next_steps.json'
sample_size = 2  # Change this to the desired sample size
formatted_text = format_examples(file_path, sample_size)

print("EXAMPLES = \"\"\"\n" + formatted_text + "\"\"\"")