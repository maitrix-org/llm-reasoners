import random
from itertools import permutations
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass
import os
from datetime import datetime
import random
import gen_proof as gen_proof
import csv
import argparse
import json

@dataclass
class Constraint:
    text: str              # Human-readable constraint
    object1: Optional[int]   # First object's encoded value (if applicable)
    object2: Optional[int]   # Second object's encoded value (if applicable)
    constraint_type: str   # Type of mathematical constraint
    value: Optional[int]   # Additional value needed for constraint (e.g., number of birds between)
    
    def evaluate(self, positions: List[int]) -> bool:
        """Evaluate if the constraint is satisfied for given positions."""
        if self.constraint_type == "left_of":
            return positions.index(self.object1) < positions.index(self.object2)
        elif self.constraint_type == "right_of":
            return positions.index(self.object1) > positions.index(self.object2)
        elif self.constraint_type == "adjacent_left":
            return positions.index(self.object1) + 1 == positions.index(self.object2)
        elif self.constraint_type == "adjacent_right":
            return positions.index(self.object1) - 1 == positions.index(self.object2)
        elif self.constraint_type == "birds_between":
            return abs(positions.index(self.object1) - positions.index(self.object2)) - 1 == self.value
        elif self.constraint_type == "absolute_position_left":
            return positions.index(self.object1) == self.value - 1
        elif self.constraint_type == "absolute_position_right":
            return positions.index(self.object1) == len(positions) - self.value
        return False

class BirdPuzzleGenerator:
    def __init__(self, objects: List[str]):
        self.birds = objects
        self.num_positions = len(objects)
        # Encode birds as integers (0 to n-1)
        self.bird_encoding = {bird: i for i, bird in enumerate(objects)}
        self.bird_decoding = {i: bird for i, bird in enumerate(objects)}
        
        # Generate ground truth
        self.ground_truth_birds = list(objects)
        random.shuffle(self.ground_truth_birds)
        # Store encoded ground truth positions
        self.ground_truth = [self.bird_encoding[bird] for bird in self.ground_truth_birds]
        
    def generate_all_constraints(self) -> List[Constraint]:
        """Generate all possible true constraints based on ground truth."""
        absolute_constraints = []
        relative_constraints = []
        # Generate all types of constraints
        for i in range(self.num_positions):
            bird1_code = self.ground_truth[i]
            bird1_name = self.bird_decoding[bird1_code]
            
            # Absolute position constraints
            pos_suffix = lambda n: 'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
            left_pos = i + 1
            right_pos = self.num_positions - i
            
            absolute_constraints.append(Constraint(
                text=f"The {bird1_name} is the {left_pos}{pos_suffix(left_pos)} from the left",
                object1=bird1_code,
                object2=None,
                constraint_type="absolute_position_left",
                value=left_pos
            ))
            
            absolute_constraints.append(Constraint(
                text=f"The {bird1_name} is the {right_pos}{pos_suffix(right_pos)} from the right",
                object1=bird1_code,
                object2=None,
                constraint_type="absolute_position_right",
                value=right_pos
            ))
            
            # Relative position constraints
            for j in range(i + 1, self.num_positions):
                bird2_code = self.ground_truth[j]
                bird2_name = self.bird_decoding[bird2_code]
                
                # Left/Right relationships
                relative_constraints.append(Constraint(
                    text=f"The {bird1_name} is to the left of the {bird2_name}",
                    object1=bird1_code,
                    object2=bird2_code,
                    constraint_type="left_of",
                    value=None
                ))
                
                relative_constraints.append(Constraint(
                    text=f"The {bird2_name} is to the right of the {bird1_name}",
                    object1=bird2_code,
                    object2=bird1_code,
                    constraint_type="right_of",
                    value=None
                ))
                
                # Birds between
                birds_between = j - i - 1
                if birds_between > 0:
                    plural = "s" if birds_between > 1 else ""
                    relative_constraints.append(Constraint(
                        text=f"There {['is', 'are'][birds_between > 1]} {birds_between} bird{plural} between the {bird1_name} and the {bird2_name}",
                        object1=bird1_code,
                        object2=bird2_code,
                        constraint_type="birds_between",
                        value=birds_between
                    ))
                
                # Adjacent relationships
                if birds_between == 0:
                    relative_constraints.append(Constraint(
                        text=f"The {bird1_name} is immediately to the left of the {bird2_name}",
                        object1=bird1_code,
                        object2=bird2_code,
                        constraint_type="adjacent_left",
                        value=None
                    ))
                    
                    relative_constraints.append(Constraint(
                        text=f"The {bird2_name} is immediately to the right of the {bird1_name}",
                        object1=bird2_code,
                        object2=bird1_code,
                        constraint_type="adjacent_right",
                        value=None
                    ))
        
        return absolute_constraints, relative_constraints

    def verify_arrangement(self, constraints: List[Constraint], test_arrangement: List[int]) -> bool:
        """Verify if a given arrangement satisfies all constraints using mathematical evaluation."""
        return all(constraint.evaluate(test_arrangement) for constraint in constraints)

    def find_minimal_constraints(self) -> List[Constraint]:
        """Find a minimal set of constraints that uniquely determines the ground truth.
        Uses incremental constraint addition until solution space reduces to one arrangement."""
        absolute_constraints, relative_constraints = self.generate_all_constraints()
        # Shuffle constraints to randomize selection order
        abs_constraints = list(absolute_constraints)
        random.shuffle(abs_constraints)
        rel_constraints = list(relative_constraints)
        random.shuffle(rel_constraints)
        
        selected_constraints = []

        # num_absolute = random.randint(0, int(self.num_positions*0.4))
        num_absolute = 0
        while len(selected_constraints) < num_absolute:
            selected_constraints.append(abs_constraints.pop(0))
        
        while rel_constraints and len(selected_constraints) < len(rel_constraints):
            # Take the next constraint
            current_constraint = rel_constraints.pop(0)
            selected_constraints.append(current_constraint)
            
            # Find all valid arrangements given current constraints
            valid_arrangements = []
            for perm in permutations(range(self.num_positions)):
                if self.verify_arrangement(selected_constraints, list(perm)):
                    valid_arrangements.append(list(perm))
                    # if len(valid_arrangements) > 1:
                    #     break
            
            # If we've found a unique solution matching ground truth
            if len(valid_arrangements) == 1:
                if valid_arrangements[0] == self.ground_truth:
                    print("Found minimal constraint set!")
                    return selected_constraints
                else:
                    # Our constraints led to a wrong unique solution
                    print("Warning: Constraints led to incorrect unique solution.")
                    # Remove the last constraint and try a different one
                    selected_constraints.pop()
                    continue
            
            # If we have no valid arrangements, the last constraint was incompatible
            if len(valid_arrangements) == 0:
                print("Warning: No valid arrangements with these constraints.")
                # Remove the incompatible constraint and continue
                selected_constraints.pop()
                continue
        
        print("Warning: Could not find a minimal constraint set.")
        return selected_constraints

    def generate_puzzle(self) -> Tuple[List[Constraint], List[List[int]]]:
        """Generate a puzzle with constraints and wrong arrangements."""
        constraints = self.find_minimal_constraints()
        return constraints

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ordering puzzles for evaluating reasoning in LLMs")
    parser.add_argument("--num_puzzles", type=int, default=10000, help="Number of puzzles to generate")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory for generated puzzles")
    parser.add_argument("--output_file", type=str, default="puzzles_dataset.json", help="Output filename for generated puzzles")
    parser.add_argument("--test", type=bool, default=False, help="Generate set without solution traces")
    args = parser.parse_args()

    birds = ["hawk", "hummingbird", "quail", "owl", "crow", "robin", "cardinal", "sparrow", "bluejay", "pigeon", 
            "eagle", "falcon", "finch", "woodpecker", "duck", "goose", "swan", "seagull", "pelican", "flamingo", "parrot", 
            "macaw", "peacock", "penguin", "ostrich", "emu", "vulture", "raven", "magpie", "starling", "warbler", "chickadee", 
            "swallow", "martin", "oriole", "dove", "pheasant", "turkey", "toucan", "kingfisher", "heron", "egret", "stork", 
            "kiwi", "albatross", "condor"]

    mammals = ["lion", "tiger", "bear", "wolf", "fox", "deer", "elephant", "giraffe", "zebra", "rhino", 
            "hippo", "monkey", "gorilla", "chimpanzee", "kangaroo", "koala", "panda", "raccoon", "squirrel", 
            "rabbit", "mouse", "rat", "bat", "whale", "dolphin", "seal", "walrus", "otter", "beaver", 
            "hedgehog", "armadillo", "sloth", "opossum", "weasel", "badger", "skunk", "moose", "bison", 
            "camel", "llama", "alpaca", "horse", "donkey", "cow", "goat", "sheep", "pig"]
    
    fruits = ["apple", "orange", "banana", "grape", "strawberry", "blueberry", "raspberry", "blackberry", 
            "watermelon", "cantaloupe", "honeydew", "pineapple", "mango", "papaya", "kiwi", "pear", 
            "peach", "plum", "apricot", "cherry", "lemon", "lime", "grapefruit", "coconut", "fig", 
            "date", "guava", "passion fruit", "pomegranate", "dragonfruit", "lychee", "persimmon", 
            "avocado", "starfruit", "durian", "jackfruit", "nectarine", "tangerine", "clementine"]

    vehicles = ["car", "truck", "bus", "motorcycle", "scooter", "bicycle", "train", "tram", "subway", 
                "helicopter", "airplane", "jet", "rocket", "spaceship", "submarine", "ship", "yacht", 
                "sailboat", "speedboat", "canoe", "kayak", "van", "ambulance", "firetruck", "tractor", 
                "bulldozer", "crane", "forklift", "excavator", "tank", "snowmobile", "golf cart", 
                "limousine", "taxi", "trolley", "rickshaw", "hovercraft", "blimp", "hot air balloon"]
    
    instruments = ["guitar", "piano", "violin", "cello", "bass", "harp", "flute", "clarinet", "saxophone", 
                "trumpet", "trombone", "tuba", "oboe", "bassoon", "piccolo", "recorder", 
                "harmonica", "accordion", "banjo", "mandolin", "ukulele", "drums", "bongos", "congas", 
                "tambourine", "xylophone", "marimba", "vibraphone", "glockenspiel", "triangle", "cymbals", 
                "bagpipes", "sitar", "shamisen", "erhu", "theremin", "synthesizer", "organ", "harpsichord"]

    gemstones = ["diamond", "ruby", "emerald", "sapphire", "amethyst", "topaz", "opal", "pearl", "jade", 
                "garnet", "aquamarine", "turquoise", "moonstone", "amber", "onyx", "quartz", "citrine", 
                "peridot", "tanzanite", "agate", "alexandrite", "beryl", "malachite", "obsidian", "jasper", 
                "zircon", "spinel", "sunstone", "tourmaline", "lapis lazuli", "hematite", "pyrite", 
                "rhodochrosite", "fluorite", "calcite", "azurite", "chrysoberyl", "sodalite", "carnelian"]

    people = ["Emma", "Liam", "Wei", "Priya", "Mohammed", "Sofia", "Jamal", "Yuki", 
                "Isabella", "Raj", "Noah", "Ling", "Santiago", "Amara", "Daniel", 
                "Fatima", "Chen", "Miguel", "Aisha", "Sanjay", "Harper", "Takashi", 
                "Kwame", "Elena", "Omar", "Zoe", "Hiroshi", "Nia", "Gabriel", "Jin", 
                "Maya", "Mateo", "Layla", "Arjun", "Clara", "Ahmed", "Mei", "Luis", 
                "Imani", "Alejandro", "Divya", "Kenji", "Chioma", "Paulo", "Ananya", 
                "Hassan", "Valentina", "Kofi", "Sakura", "Amir"]

    random.seed(42)
    num_puzzles = args.num_puzzles
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the "generated_QAs" folder at the same level as the Python file if it doesn't exist
    # output_folder = os.path.join(base_dir, args.output_dir)
    output_folder = args.output_dir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, args.output_file)
    
    if args.test:
        output_file_test = os.path.join(output_folder, args.output_file.replace(".json", "_test.json"))
    else:
        output_file_test = os.path.join(output_folder, args.output_file.replace(".json", "_unsolved.json"))
    
    puzzles_data = []
    puzzles_test_data = []
    object_types = {
        'birds': birds,
        'mammals': mammals,
        'fruits': fruits,
        'vehicles': vehicles, 
        'instruments': instruments,
        'gemstones': gemstones,
        'people': people
    }
    for id in range(num_puzzles):
        num_objects = random.randint(3, 7)
        
        type_name = random.choice(list(object_types.keys()))
        type_objects = object_types[type_name]
        curr_birds = random.sample(type_objects, num_objects)
        birds_str = ", ".join(curr_birds)
        instruction = f"Solve the following puzzle to determine the order of the {type_name} from left to right. The {type_name} are {birds_str}."
        generator = BirdPuzzleGenerator(curr_birds)
        
        constraints = generator.generate_all_constraints()
        constraints = generator.generate_puzzle()
        input = ""
        input_constraints = []
        for i, constraint in enumerate(constraints, 1):
            input_constraints.append(constraint.text)
            input += f"{i}. {constraint.text}\n"
        if not args.test:
            output = gen_proof.solve_puzzle(ground_truth = generator.ground_truth_birds, raw_constraints = input_constraints)
            if output:
                output = "\n".join(output)
                puzzle_data = {'id': id, 'instruction': instruction, 'input': input, 'output': output, 'ground_truth': generator.ground_truth_birds, 'num_objects': num_objects}
                puzzles_data.append(puzzle_data)
            else:
                puzzle_data = {'id': id, 'instruction':instruction, 'input': input, 'ground_truth': generator.ground_truth_birds, 'num_objects': num_objects}
                puzzles_test_data.append(puzzle_data)
        else:
            puzzle_data = {'id': id, 'instruction':instruction, 'input': input, 'ground_truth': generator.ground_truth_birds, 'num_objects': num_objects}
            puzzles_test_data.append(puzzle_data)
    if args.test:
        with open(output_file, 'w') as json_file:
            json.dump(puzzles_test_data, json_file, indent=2)
    else:
        with open(output_file, 'w') as json_file:
            json.dump(puzzles_data, json_file, indent=2)
        with open(output_file_test, 'w') as json_file_test:
            json.dump(puzzles_test_data, json_file_test, indent=2)

# example usage
# python puzzle_gen.py --num_puzzles 10000 --output_dir data/puzzles_dataset --output_file puzzles_dataset.json  --test True

