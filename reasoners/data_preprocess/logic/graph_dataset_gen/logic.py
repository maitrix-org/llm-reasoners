import os
import random
import sysconfig
from io import StringIO
from faker import Faker
from pybind11.__main__ import print_includes
import math
import argparse
import json

def build_module(name):
	import sys
	old_stdout = sys.stdout
	try:
		sys.stdout = StringIO()
		print_includes()
		includes = sys.stdout.getvalue().strip()
		sys.stdout.close()
		sys.stdout = old_stdout
	except Exception as e:
		raise e
	finally:
		sys.stdout = old_stdout

	python_extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")

	if sys.platform == "darwin":
		# macOS command
		command = (
			f"g++ -std=c++11 -Ofast -DNDEBUG -fno-stack-protector "
			f"-Wall -Wpedantic -undefined dynamic_lookup -shared -fPIC "
			f"{includes} -I. {name}.cpp -o {name}{python_extension_suffix}"
		)
	else:
		# Non-macOS command
		command = (
			f"g++ -Ofast -std=c++11 -DNDEBUG -fno-stack-protector "
			f"-Wall -Wpedantic -shared -fPIC "
			f"{includes} -I. {name}.cpp -o {name}{python_extension_suffix}"
		)

	print(command)
	if os.system(command) != 0:
		print(f"ERROR: Unable to compile `{name}.cpp`.")
		sys.exit(1)

try:
	from os.path import getmtime
	from importlib.util import find_spec

	generator_spec = find_spec('generator')
	if generator_spec == None:
		raise ModuleNotFoundError
	if getmtime(generator_spec.origin) < getmtime('generator.cpp'):
		print("C++ module `generator` is out-of-date. Compiling from source...")
		build_module("generator")
	import generator
except ModuleNotFoundError:
	print("C++ module `generator` not found. Compiling from source...")
	build_module("generator")
	import generator
except ImportError:
	print("Error loading C++ module `generator`. Compiling from source...")
	build_module("generator")
	import generator
print("C++ module `generator` loaded.")


def generate_graph_text(
	max_input_size,
	num_vertices,
	max_num_parents,
	lookahead,
	num_paths,
):

	# Get graph using the C++ generator
	inputs, outputs, labels, num_collisions = generator.generate_training_set(
		max_input_size,
		1,	# batch size
		lookahead,
		max_num_parents * num_vertices, # max edges
		set(), # reserved vertices
		-1, # distance from start
		0, # max_prefix_vertices
		True, # quiet
		num_paths # number of paths
	)

	# If the random DAG fails to generate a valid example, return None
	if inputs is None or outputs is None:
		return None

	# Token IDs
	PADDING_TOKEN      = (max_input_size - 5) // 3 + 3
	EDGE_PREFIX_TOKEN  = (max_input_size - 5) // 3 + 2
	PATH_PREFIX_TOKEN  = (max_input_size - 5) // 3 + 1
	QUERY_PREFIX_TOKEN = (max_input_size - 5) // 3 + 4

	# Function for mapping tokens to letters
	def token_to_str(token):
		if token == EDGE_PREFIX_TOKEN:
			return "E"
		elif token == QUERY_PREFIX_TOKEN:
			return "Q"
		elif token == PATH_PREFIX_TOKEN:
			return "P"
			# return ""
		elif token == PADDING_TOKEN:
			return ""
		else:
			return str(token)

	# Convert all tokens to letters
	text_tokens = [token_to_str(t) for t in inputs[0] if token_to_str(t) != ""]

	final_str = " ".join(text_tokens)
	return final_str, labels[0]


def generate_name(n):
	"""
	Generates n random names using the Faker library.
	"""
	fake = Faker()
	# return [fake.unique.first_name() for i in range(n)]
	return [fake.unique.name() for i in range(n)]


def random_syllable():
	"""
	Generates a random syllable based on a randomly chosen pattern.
	The pattern is built from 'C' (consonant) and 'V' (vowel).
	"""
	vowels = "aeiou"
	consonants = "bcdfghjklmnpqrstvwxyz"
	# Syllable patterns (e.g. "CV" means a consonant followed by a vowel)
	patterns = ["CV", "VC", "CVC", "CVV", "CCV", "VCV", "VCC"]
	pattern = random.choice(patterns)

	syllable = ""
	for char in pattern:
		if char == "C":
			syllable += random.choice(consonants)
		elif char == "V":
			syllable += random.choice(vowels)
	return syllable


def generate_fake_noun():
	"""
	Generates a fake noun by concatenating two randomly generated syllables.
	"""
	return random_syllable() + random_syllable()


def generate_fake_nouns(n):
	"""
	Returns a list of n fake nouns.
	"""
	nouns = set()
	while len(nouns) < n:
		nouns.add(generate_fake_noun())
	return list(nouns)


# def generate_words(n):
# 	fake = Faker()
# 	return fake.words(nb=n)


def logic_paragraph_from_tokens(tokens_str: str, next_step: int, use_diff_names=True):
	# Split string into tokens
	tokens = tokens_str.strip().split()

	edges = []
	queries = []

	# Keep track of nodes in a set
	node_ids = set()

	i = 0
	while i < len(tokens):
		t = tokens[i]

		if t == "E":
			# Expect "E A B"
			if i + 2 >= len(tokens):
				break  # malformed
			A = int(tokens[i + 1])
			B = int(tokens[i + 2])
			edges.append((A, B))
			node_ids.update([A, B])
			i += 3

		elif t == "Q":
			# Expect "Q X Y"
			if i + 2 >= len(tokens):
				break
			X = int(tokens[i + 1])
			Y = int(tokens[i + 2])
			queries.append((X, Y))
			node_ids.update([X, Y])
			i += 3

		elif t == "P":
			# "P" is just a path prefix token; ignore
			i += 1

		else:
			# Possibly a stray token or something else (like a vertex ID alone).
			i += 1

	# Generate fake names and fake adjectives
	num_nodes = len(node_ids)
	all_names = generate_name(num_nodes)
	all_adjs = generate_fake_nouns(num_nodes)

	sorted_nodes = sorted(node_ids)
	id_to_pair = {}  # node_id -> (name, adjective)
	# Assign all nodes the same name if use_diff_names == False
	for idx, node_id in enumerate(sorted_nodes):
		id_to_pair[node_id] = (all_names[idx if use_diff_names else 0], all_adjs[idx])

	# Lines of logic
	lines = ["Given the following list of predicates:"]

	def get_logic_line(name_a: str, name_b: str, adj_a: str, adj_b: str) -> str:
		choices = [
			f"If {name_a} is {adj_a}, then {name_b} is {adj_b}.",
			f"{name_a} is {adj_a} implies {name_b} is {adj_b}.",
			f"{name_b} is {adj_b} is true if {name_a} is {adj_a}.",
			# f"{adj_b} is true if {adj_a} is true.",
			# f"If {adj_a} then {adj_b} is true.",
			# f"If {adj_a} is true then {adj_b}.",
			f"Given {name_a} is {adj_a} then {name_b} is {adj_b}.",
		]

		sentence = random.choice(choices)
		return sentence[0].upper() + sentence[1:]

	# For each edge:  E A B => "If name(A) is adj(A), then name(A) is adj(B)."
	for (A, B) in edges:
		name_A, adj_A = id_to_pair[A]
		name_B, adj_B = id_to_pair[B]
		lines.append(get_logic_line(name_A, name_B, adj_A, adj_B))

	# For each query: Q X Y => "If name(X) is adj(X), prove that name(X) is adj(Y)."
	for (X, Y) in queries:
		name_x, adj_x = id_to_pair[X]
		name_y, adj_y = id_to_pair[Y]
		lines.append(f"\n\nIf {name_x} is {adj_x}, what is the next step to prove that {name_y} is {adj_y}?")

	# Join all lines into one paragraph
	paragraph = " ".join(lines)

	# Get the correct adjective for the next step
	next_step_adj = id_to_pair[next_step]
	next_step_adj = id_to_pair[next_step][1]

	return paragraph, next_step_adj



def main(samples_per_test: int = 3, lookahead_range: list = range(1, 5), num_paths: int = 2, max_num_parents: int = 3, logic: bool = False, seed: int = None, verbose: bool = True, print_prompts: bool = False, output_file:str ="graph_search_data.json", model: str = "gpt-4o"):
	if seed is not None:
		random.seed(seed)
		generator.set_seed(seed)

	graph_search_puzzles = []
	correct_responses = []
	look_ahead_values = []
	counter = -1
	for look_ahead in lookahead_range:
		for _ in range(samples_per_test):
			txt, next_step = generate_graph_text(
				max_input_size=max_num_parents * look_ahead * num_paths * 4,
				num_vertices=max_num_parents * look_ahead,
				max_num_parents=max_num_parents,
				lookahead=look_ahead,
				num_paths=num_paths,
			)
			counter += 1
			# Create the prompt for the graph search
			prompt = (f"{txt}\nAbove is a representation of a directed graph search problem, "
					  f"where E A B represents an edge from A to B, and Q X Y represents starting from X and ending at Y, "
					  f"find the shortest path. The vertex after P indicates our current position. Respond with only the "
					  f"next vertex on the shortest path from X to Y and nothing else.")

			# Change the prompt to a logic puzzle if the logic option is enabled
			if logic:
				logic, next_step_adj = logic_paragraph_from_tokens(txt, next_step)
				prompt = f"{logic} Respond with only the trait of the next step."

			
			correct_response = next_step_adj if logic else next_step
			puzzle = {
				"id": counter,
				"prompt": prompt,
				"correct_response": correct_response,
				"look_ahead": look_ahead
			}
			graph_search_puzzles.append(puzzle)
			if print_prompts:
				print(f"Prompt: {prompt}\n")
				print(f"Correct:   {next_step_adj if logic else next_step}\n")

	with open(output_file, "w") as json_file:
		json.dump(graph_search_puzzles, json_file, indent=2)
	

def multiplicative_range(start, stop, step):
	result = []
	current = start
	while (step > 1 and current < stop) or (step < 1 and current > stop):
		result.append(current)
		current *= step
	return result

def exponential_range(base, start, end):
	res = [round(base ** n) for n in range(start, end + 1)]
	res = list(set(res))
	return sorted(res)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate graph search problems.")
	parser.add_argument("--output_dir", type=str, default="../../../data/raw", help="output directory for generated problems")
	parser.add_argument("--output_file", type=str, default="graph_search.json", help="output file for generated problems")
	parser.add_argument("--num_samples", type=int, default=10000, help="number of samples to generate")
	args = parser.parse_args()
	# Set the output directory and file
	# base_dir = os.path.dirname(os.path.abspath(__file__))
	output_dir = args.output_dir
	# Create the output directory if it doesn't exist
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	output_file = os.path.join(output_dir, args.output_file)
	# Set the random seed for reproducibility
	seed = 42
	random.seed(seed)
	look_ahead = exponential_range(1.3, 1, 24)
	print(f"look_ahead range={look_ahead}")
	samples_per_test =  round(args.num_samples/len(look_ahead))
	print(f"samples_per_test={samples_per_test}")
	main(
		samples_per_test=samples_per_test,
		lookahead_range=look_ahead,
		num_paths=9,
		logic=True,
		verbose=False,
		print_prompts=False,
		output_file = output_file,
		seed=seed,
	)

# example usage:
# python graph_search.py --output_dir ../data/graph_dataset --output_file graph_search.json --num_samples 10000

