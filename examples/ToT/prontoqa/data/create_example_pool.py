import json

def extract_unique_in_context_examples(input_file, output_file):
    # Read the original JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Set for tracking unique examples and test questions
    unique_examples = set()
    test_questions = set()

    # First, collect all test questions
    for key, value in data.items():
        if "test_example" in value:
            test_question = json.dumps(value["test_example"]["question"], sort_keys=True)
            test_questions.add(test_question)

    # List to store unique in-context examples
    example_pool = []

    # Extract unique in-context examples
    for key, value in data.items():
        for example_key, example_value in value.items():
            if 'in_context_example' in example_key:
                # Convert the example to a JSON string for comparison
                example_str = json.dumps(example_value["question"], sort_keys=True)
                if example_str not in unique_examples and example_str not in test_questions:
                    unique_examples.add(example_str)
                    example_pool.append(example_value)

    # Create the final dictionary with the example pool
    print(len(example_pool))
    final_data = {"example_pool": example_pool}

    # Write the extracted data to a new JSON file
    with open(output_file, 'w') as file:
        json.dump(final_data, file, indent=4)


# Usage
input_file = '345hop_random_true.json'
output_file = 'example_pool.json'
extract_unique_in_context_examples(input_file, output_file)
