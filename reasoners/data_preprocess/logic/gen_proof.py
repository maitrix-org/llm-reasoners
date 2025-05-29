
import os
from typing import List, Set, Dict, Tuple, Optional
import re
from dataclasses import dataclass, field
from copy import deepcopy
import random
import sys
import pdb
import signal
import time
from contextlib import contextmanager


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

DEBUG = False

@dataclass
class SearchNode:
    """Data class to represent a search node in the tree."""
    depth: int
    arrangement: Dict[str, int] = field(default_factory=dict)
    satisfied_constraints: Set[int] = field(default_factory=set)
    children: List['SearchNode'] = field(default_factory=list)
    is_leaf: bool = False
    parent: Optional['SearchNode'] = None
    

    def __str__(self):
        positions = [-1] * len(self.arrangement)
        for bird, pos in self.arrangement.items():
            if pos != -1:
                positions[pos] = bird
        return f"Arrangement: {positions}, Satisfied: {self.satisfied_constraints}"

class PuzzleSolver:
    def __init__(self, birds: List[str], puzzle_constraints: List[str]=None):
        self.birds = birds
        self.num_positions = len(birds)
        self.raw_constraints = puzzle_constraints
        self.constraints: List[Constraint] = []
        self.absolute_constraints: List[Constraint] = []
        self.relative_constraints: List[Constraint] = []
        self.steps: List[str] = []
        self.ground_truth: List[str] = []
        
    
    def add_step(self, step: str):
        """Add a step to the list of reasoning steps."""
        self.steps.append(step)
        # print(step)



    def parse_constraints(self):
        """Parse text constraints into structured Constraint objects."""
        for text in self.raw_constraints:
            constraint = self._parse_single_constraint(text)
            self.constraints.append(constraint)
            
            # Separate absolute and relative constraints
            if constraint.constraint_type in ['absolute_position_left', 'absolute_position_right']:
                self.absolute_constraints.append(constraint)
            else:
                self.relative_constraints.append(constraint)
    
    def _parse_single_constraint(self, text:str)->Constraint:
        """Parse a single text constraint into a Constraint object."""
        # Pattern for extracting bird names
        bird_pattern = '|'.join(self.birds)
        # Absolute position patterns
        if "from the left" in text or "from the right" in text:
            match = re.search(rf"The ({bird_pattern}) is the (\d+)[a-z]+ from the (left|right)", text)
            if match:
                bird, position, direction = match.groups()
                return Constraint(
                    text=text,
                    bird1=bird,
                    bird2=None,
                    constraint_type=f"absolute_position_{direction}",
                    value=int(position)
                )
        
        # Birds between pattern
        if "bird" in text and "between" in text:
            match = re.search(rf"There (?:is|are) (\d+) birds? between the ({bird_pattern}) and the ({bird_pattern})", text)
            if match:
                num_birds, bird1, bird2 = match.groups()
                return Constraint(
                    text=text,
                    bird1=bird1,
                    bird2=bird2,
                    constraint_type="birds_between",
                    value=int(num_birds)
                )
        
        # Immediately adjacent patterns
        if "immediately" in text:
            if "to the left of" in text:
                match = re.search(f"The ({bird_pattern}) is immediately to the left of the ({bird_pattern})", text)
                bird1, bird2 = match.groups()
                return Constraint(
                    text=text,
                    bird1=bird1,
                    bird2=bird2,
                    constraint_type="adjacent_left",
                    value=None
                )
            elif "to the right of" in text:
                match = re.search(f"The ({bird_pattern}) is immediately to the right of the ({bird_pattern})", text)
                bird1, bird2 = match.groups()
                return Constraint(
                    text=text,
                    bird1=bird1,
                    bird2=bird2,
                    constraint_type="adjacent_right",
                    value=None
                )
        
        # General left/right patterns
        if "to the left of" in text:
            match = re.search(f"The ({bird_pattern}) is to the left of the ({bird_pattern})", text)
            bird1, bird2 = match.groups()
            return Constraint(
                text=text,
                bird1=bird1,
                bird2=bird2,
                constraint_type="left_of",
                value=None
            )
        elif "to the right of" in text:
            match = re.search(f"The ({bird_pattern}) is to the right of the ({bird_pattern})", text)
            bird1, bird2 = match.groups()
            return Constraint(
                text=text,
                bird1=bird1,
                bird2=bird2,
                constraint_type="right_of",
                value=None
            )
        
        raise ValueError(f"Could not parse constraint: {text}")
    


        
    def init_helper(self, file_path: str):
        """Initialize the solver with the puzzle file."""
        self.ground_truth = self.read_file(file_path)
        self.birds = random.shuffle(self.ground_truth)# Shuffle the birds
        self.parse_constraints()

    def verify_constraint(self, arrangement: Dict[str, int], constraint: Constraint) -> bool:
        """Check if a single constraint is satisfied by current arrangement"""
        """Return true only if the constraint is satisfied else false"""
        if constraint.constraint_type == "absolute_position_left":
            pos = arrangement.get(constraint.bird1, -1)
            return pos == constraint.value - 1
            
        elif constraint.constraint_type == "absolute_position_right":
            pos = arrangement.get(constraint.bird1, -1)
            return pos == self.num_positions - constraint.value if pos != -1 else False
            
        elif constraint.constraint_type == "adjacent_left":
            pos1 = arrangement.get(constraint.bird1, -1)
            pos2 = arrangement.get(constraint.bird2, -1)
            return pos1 != -1 and pos2 != -1 and pos1 + 1 == pos2
            
        elif constraint.constraint_type == "adjacent_right":
            pos1 = arrangement.get(constraint.bird1, -1)
            pos2 = arrangement.get(constraint.bird2, -1)
            return pos1 != -1 and pos2 != -1 and pos1 - 1 == pos2
            
        elif constraint.constraint_type == "birds_between":
            pos1 = arrangement.get(constraint.bird1, -1)
            pos2 = arrangement.get(constraint.bird2, -1)
            if pos1 != -1 and pos2 != -1:
                return abs(pos1 - pos2) - 1 == constraint.value
            return False  # Can't verify yet
            
        elif constraint.constraint_type == "left_of":
            pos1 = arrangement.get(constraint.bird1, -1)
            pos2 = arrangement.get(constraint.bird2, -1)
            return pos1 < pos2 if pos1 != -1 and pos2 != -1 else False
            
        elif constraint.constraint_type == "right_of":
            pos1 = arrangement.get(constraint.bird1, -1)
            pos2 = arrangement.get(constraint.bird2, -1)
            return pos1 > pos2 if pos1 != -1 and pos2 != -1 else False
            
        return False
    
    def solve(self) -> Optional[Dict[str, int]]:
        """Main solving function"""
        # Parse constraints if not done already
        if not self.constraints:
            self.parse_constraints()
            
        # Start with empty arrangement
        initial_arrangement = {bird: -1 for bird in self.birds}
        # First apply absolute constraints
        self.add_step("Starting with absolute position constraints:")
        for i, constraint in enumerate(self.absolute_constraints):
            bird = constraint.bird1
            if constraint.constraint_type == "absolute_position_left":
                initial_arrangement[bird] = constraint.value - 1
                self.add_step(f"Placing {bird} at position {constraint.value} from left")
            else:  # absolute_position_right
                pos = self.num_positions - constraint.value
                initial_arrangement[bird] = pos
                self.add_step(f"Placing {bird} at position {pos + 1} from left ({constraint.value} from right)")
        
        # Create initial node
        initial_node = SearchNode(
            arrangement=initial_arrangement,
            satisfied_constraints=set(),
            depth=0
        )
        
        # Start Tree Search
        self.add_step("Starting search to find complete arrangement...")
        result = self.dfs(initial_node)
        
        if result:
            positions = [-1] * self.num_positions
            for bird, pos in result.arrangement.items():
                positions[pos] = bird
            if -1 in positions:
                index = positions.index(-1)
                positions[index] = [bird for bird in self.birds if bird not in positions][0]
                self.add_step(f"{positions[index]} is placed at position {index} since only one spot was left")
            self.add_step("Found solution!")
            self.add_step(f"Final arrangement: {positions}")
            return result.arrangement
        else:
            self.add_step("\nNo solution found!")
            return None
        
    def dfs(self, node: SearchNode) -> Optional[SearchNode]:
        """Depth-first search implementation"""
        # First verify all constraints are satisfied for current arrangement
        if DEBUG:
            pdb.set_trace()
            print(f"\nDEBUG: Current depth: {node.depth}")
            print(f"DEBUG: Current arrangement: {node.arrangement}")
            print(f"DEBUG: Satisfied constraints: {node.satisfied_constraints}")
        violated_constraint = None
        current_constraint = None
        current_constraint_index = None
        for i, constraint in enumerate(self.relative_constraints):
            if i not in node.satisfied_constraints:
                consistent = self.verify_constraint(node.arrangement, constraint)
                if consistent:
                    node.satisfied_constraints.add(i)
                else:
                    current_constraint = constraint
                    current_constraint_index = i
                    break
        
        # return if solution found
        if not current_constraint:
            if DEBUG: print("DEBUG: No more constraints to satisfy!")
            node.is_leaf = True
            return node
        
        filled = set(pos for pos in node.arrangement.values() if pos != -1)
        empty_positions = sorted(set(range(self.num_positions)) - filled)
        if DEBUG: print(f"DEBUG: Empty positions: {empty_positions}")
        # empty_positions = [i for bird, i in node.arrangement.items() if i == -1]
        # import pdb; pdb.set_trace()
        k = len(empty_positions)
        if k==0 and current_constraint:
            if DEBUG: print("DEBUG: No empty positions left but constraints remain!")
            return None
        current_arrangement = ["_"] * self.num_positions
        for bird, pos in node.arrangement.items():
            if pos != -1:
                current_arrangement[pos] = bird

        if current_constraint:
            if current_constraint.constraint_type == "left_of":
                if DEBUG: pdb.set_trace()
                bird1 = current_constraint.bird1
                bird2 = current_constraint.bird2
                pos1 = node.arrangement.get(bird1, -1)
                pos2 = node.arrangement.get(bird2, -1)
                if pos1 != -1 and pos2 != -1 and pos1 >= pos2:
                    violated_constraint = current_constraint
                    self.add_step(f"Current arrangement: {current_arrangement}; Constraint violated: {current_constraint.text}, cannot proceed on this path")
                    return None
                elif pos1 ==-1 and pos2 == -1:
                    for i in range(k-1):
                        for j in range(i+1, k):
                            new_node = deepcopy(node)
                            new_node.arrangement[bird1] = empty_positions[i]
                            new_node.arrangement[bird2] = empty_positions[j]
                            new_node.satisfied_constraints.add(current_constraint_index)
                            new_node.depth += 1
                            new_node.parent = node
                            node.children.append(new_node)
                            self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird1} to the left of {bird2} at positions {empty_positions[i]} and {empty_positions[j]}")
                            result = self.dfs(new_node)
                            if result:
                                return result
                            else:
                                self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} to the left of {bird2} at positions {empty_positions[i]} and {empty_positions[j]}")
                else:
                    if pos1 == -1:
                        for i in range(k):
                            if empty_positions[i] < pos2:
                                new_node = deepcopy(node)
                                new_node.arrangement[bird1] = empty_positions[i]
                                new_node.satisfied_constraints.add(current_constraint_index)
                                new_node.depth += 1
                                new_node.parent = node
                                node.children.append(new_node)
                                self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird1} to the left of {bird2} at positions {empty_positions[i]}")
                                result = self.dfs(new_node)
                                if result:
                                    return result
                                else:
                                    self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} to the left of {bird2} at positions {empty_positions[i]}")
                    elif pos2 == -1:
                        for i in range(k):
                            if empty_positions[i] > pos1:
                                new_node = deepcopy(node)
                                new_node.arrangement[bird2] = empty_positions[i]
                                new_node.satisfied_constraints.add(current_constraint_index)
                                new_node.depth += 1
                                new_node.parent = node
                                node.children.append(new_node)
                                self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird2} to the right of {bird1} at positions {empty_positions[i]}")
                                result = self.dfs(new_node)
                                if result:
                                    return result
                                else:
                                    self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird2} to the right of {bird1} at positions {empty_positions[i]}")
            elif current_constraint.constraint_type == "right_of":
                if DEBUG: pdb.set_trace()
                bird1 = current_constraint.bird1
                bird2 = current_constraint.bird2
                pos1 = node.arrangement.get(bird1, -1)
                pos2 = node.arrangement.get(bird2, -1)
                if pos1 != -1 and pos2 != -1 and pos1 <= pos2:
                    violated_constraint = current_constraint
                    self.add_step(f"Current arrangement: {current_arrangement}; Constraint violated: {current_constraint.text}, cannot proceed on this path")
                    return None
                elif pos1 == -1 and pos2 == -1:
                    for i in range(k-1):
                        for j in range(i+1, k):
                            new_node = deepcopy(node)
                            new_node.arrangement[bird1] = empty_positions[j]
                            new_node.arrangement[bird2] = empty_positions[i]
                            new_node.satisfied_constraints.add(current_constraint_index)
                            new_node.depth += 1
                            new_node.parent = node
                            node.children.append(new_node)
                            self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird1} to the right of {bird2} at positions {empty_positions[j]} and {empty_positions[i]}")
                            result = self.dfs(new_node)
                            if result:
                                return result
                            else:
                                self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} to the right of {bird2} at positions {empty_positions[j]} and {empty_positions[i]}")
                else:
                    if pos1 == -1:
                        for i in range(k):
                            if empty_positions[i] > pos2:
                                new_node = deepcopy(node)
                                new_node.arrangement[bird1] = empty_positions[i]
                                new_node.satisfied_constraints.add(current_constraint_index)
                                new_node.depth += 1
                                new_node.parent = node
                                node.children.append(new_node)
                                self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird1} to the right of {bird2} at positions {empty_positions[i]}")
                                result = self.dfs(new_node)
                                if result:
                                    return result
                                else:
                                    self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} to the right of {bird2} at positions {empty_positions[i]}")
                    elif pos2 == -1:
                        for i in range(k):
                            if empty_positions[i] < pos1:
                                new_node = deepcopy(node)
                                new_node.arrangement[bird2] = empty_positions[i]
                                new_node.satisfied_constraints.add(current_constraint_index)
                                new_node.depth += 1
                                new_node.parent = node
                                node.children.append(new_node)
                                self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird2} to the left of {bird1} at positions {empty_positions[i]}")
                                result = self.dfs(new_node)
                                if result:
                                    return result
                                else:
                                    self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird2} to the left of {bird1} at positions {empty_positions[i]}")
            elif current_constraint.constraint_type == "adjacent_left":
                if DEBUG: pdb.set_trace()
                bird1 = current_constraint.bird1
                bird2 = current_constraint.bird2
                pos1 = node.arrangement.get(bird1, -1)
                pos2 = node.arrangement.get(bird2, -1)
                if pos1 != -1 and pos2 != -1 and pos1 + 1 != pos2:
                    violated_constraint = current_constraint
                    self.add_step(f"Current arrangement: {current_arrangement}; Constraint violated: {current_constraint.text}, cannot proceed on this path")
                    return None
                elif pos1 == -1 and pos2 == -1:
                    for i in range(k-1):
                        for j in range(i+1, k):
                            if empty_positions[i] + 1 == empty_positions[j]:
                                new_node = deepcopy(node)
                                new_node.arrangement[bird1] = empty_positions[i]
                                new_node.arrangement[bird2] = empty_positions[j]
                                new_node.satisfied_constraints.add(current_constraint_index)
                                new_node.depth += 1
                                new_node.parent = node
                                node.children.append(new_node)
                                self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird1} immediately to the left of {bird2} at positions {empty_positions[i]} and {empty_positions[j]}")
                                result = self.dfs(new_node)
                                if result:
                                    return result
                                else:
                                    self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} immediately to the left of {bird2} at positions {empty_positions[i]} and {empty_positions[j]}")
                else:
                    if pos1 == -1:
                        if pos2-1 not in empty_positions:
                            self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} immediately to the left of {bird2} at positions {pos2-1}")
                            return None
                        new_node = deepcopy(node)
                        new_node.arrangement[bird1] = pos2 - 1
                        new_node.satisfied_constraints.add(current_constraint_index)
                        new_node.depth += 1
                        new_node.parent = node
                        node.children.append(new_node)
                        self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird1} immediately to the left of {bird2} at positions {pos2-1}")
                        result = self.dfs(new_node)
                        if result:
                            return result
                        else:
                            self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} immediately to the left of {bird2} at positions {pos2-1}")
                    elif pos2 == -1:
                        if pos1+1 not in empty_positions:
                            self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} immediately to the left of {bird2} at positions {pos1+1}")
                            return None
                        new_node = deepcopy(node)
                        new_node.arrangement[bird2] = pos1 + 1
                        new_node.satisfied_constraints.add(current_constraint_index)
                        new_node.depth += 1
                        new_node.parent = node
                        node.children.append(new_node)
                        self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird2} immediately to the right of {bird1} at positions {pos1+1}")
                        result = self.dfs(new_node)
                        if result:
                            return result
                        else:
                            self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird2} immediately to the right of {bird1} at positions {pos1+1}")
            elif current_constraint.constraint_type == "adjacent_right":
                if DEBUG: pdb.set_trace()
                bird1 = current_constraint.bird1
                bird2 = current_constraint.bird2
                pos1 = node.arrangement.get(bird1, -1)
                pos2 = node.arrangement.get(bird2, -1)
                if pos1 != -1 and pos2 != -1 and pos1 - 1 != pos2:
                    violated_constraint = current_constraint
                    self.add_step(f"Current arrangement: {current_arrangement}; Constraint violated: {current_constraint.text}, cannot proceed on this path ")
                    return None
                elif pos1 == -1 and pos2 == -1:
                    for i in range(k-1):
                        for j in range(i+1, k):
                            if empty_positions[i] + 1 == empty_positions[j]:
                                new_node = deepcopy(node)
                                new_node.arrangement[bird1] = empty_positions[j]
                                new_node.arrangement[bird2] = empty_positions[i]
                                new_node.satisfied_constraints.add(current_constraint_index)
                                new_node.depth += 1
                                new_node.parent = node
                                node.children.append(new_node)
                                self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird1} immediately to the right of {bird2} at positions {empty_positions[j]} and {empty_positions[i]}")
                                result = self.dfs(new_node)
                                if result:
                                    return result
                                else:
                                    self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} immediately to the right of {bird2} at positions {empty_positions[j]} and {empty_positions[i]}")
                else:
                    if pos1 == -1:
                        if pos2+1 not in empty_positions:
                            self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} immediately to the right of {bird2} at positions {pos2+1}")
                            return None
                        new_node = deepcopy(node)
                        new_node.arrangement[bird1] = pos2 + 1
                        new_node.satisfied_constraints.add(current_constraint_index)
                        new_node.depth += 1
                        new_node.parent = node
                        node.children.append(new_node)
                        self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird1} immediately to the right of {bird2} at positions {pos2+1}")
                        result = self.dfs(new_node)
                        if result:
                            return result
                        else:
                            self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} immediately to the right of {bird2} at positions {pos2+1}")
                    elif pos2 == -1:
                        if pos1-1 not in empty_positions:
                            self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} immediately to the right of {bird2} at positions {pos1-1}")
                            return None
                        new_node = deepcopy(node)
                        new_node.arrangement[bird2] = pos1 - 1
                        new_node.satisfied_constraints.add(current_constraint_index)
                        new_node.depth += 1
                        new_node.parent = node
                        node.children.append(new_node)
                        self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird2} immediately to the left of {bird1} at positions {pos1-1}")
                        result = self.dfs(new_node)
                        if result:
                            return result
                        else:
                            self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird2} immediately to the left of {bird1} at positions {pos1-1}")
            elif current_constraint.constraint_type == "birds_between":
                if DEBUG: pdb.set_trace()
                bird1 = current_constraint.bird1
                bird2 = current_constraint.bird2
                pos1 = node.arrangement.get(bird1, -1)
                pos2 = node.arrangement.get(bird2, -1)
                value = current_constraint.value
                if pos1 != -1 and pos2 != -1 and abs(pos1 - pos2) - 1 != value:
                    violated_constraint = current_constraint
                    self.add_step(f"Current arrangement: {current_arrangement}; Constraint violated: {current_constraint.text}, cannot proceed on this path")
                    return None
                elif pos1 == -1 and pos2 == -1:
                    for p1 in empty_positions:
                        for p2 in empty_positions:
                            if p1 != p2 and abs(p1-p2) - 1 == value:
                                # 2 paths bird 1 left of bird 2 and bird 1 right of bird 2
                                new_node = deepcopy(node)
                                new_node.arrangement[bird1] = p1
                                new_node.arrangement[bird2] = p2
                                new_node.satisfied_constraints.add(current_constraint_index)
                                new_node.depth += 1
                                new_node.parent = node
                                node.children.append(new_node)
                                self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird1} to the left of {bird2} at positions {p1} and {p2}")
                                result = self.dfs(new_node)
                                if result:
                                    return result
                                else:
                                    self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} to the left of {bird2} at positions {p1} and {p2}")
                                    new_node = deepcopy(node)
                                    new_node.arrangement[bird1] = p2
                                    new_node.arrangement[bird2] = p1
                                    new_node.satisfied_constraints.add(current_constraint_index)
                                    new_node.depth += 1
                                    new_node.parent = node
                                    node.children.append(new_node)
                                    self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird1} to the right of {bird2} at positions {p2} and {p1}")
                                    result = self.dfs(new_node)
                                    if result:
                                        return result
                                    else:
                                        self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} to the right of {bird2} at positions {p2} and {p1}")
                else:
                    if pos1 == -1:
                        if pos2 - value - 1 in empty_positions:
                            new_node = deepcopy(node)
                            new_node.arrangement[bird1] = pos2 - value - 1
                            new_node.satisfied_constraints.add(current_constraint_index)
                            new_node.depth += 1
                            new_node.parent = node
                            node.children.append(new_node)
                            self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird1} to the left of {bird2} at positions {pos2 - value - 1}")
                            result = self.dfs(new_node)
                            if result:
                                return result
                            else:
                                self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} to the left of {bird2} at positions {pos2 - value - 1}")
                        if pos2 + value +1 in empty_positions:
                            new_node = deepcopy(node)
                            new_node.arrangement[bird1] = pos2 + value + 1
                            new_node.satisfied_constraints.add(current_constraint_index)
                            new_node.depth += 1
                            new_node.parent = node
                            node.children.append(new_node)
                            self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird1} to the right of {bird2} at positions {pos2 + value + 1}")
                            result = self.dfs(new_node)
                            if result:
                                return result
                            else:
                                self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird1} to the right of {bird2} at positions {pos2 + value + 1}")
                        else:
                            return None
                    elif pos2 == -1:
                        if DEBUG: pdb.set_trace()
                        if pos1 - value - 1 in empty_positions:
                            new_node = deepcopy(node)
                            new_node.arrangement[bird2] = pos1 - value - 1
                            new_node.satisfied_constraints.add(current_constraint_index)
                            new_node.depth += 1
                            new_node.parent = node
                            node.children.append(new_node)
                            self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird2} to the left of {bird1} at positions {pos1 - value - 1}")
                            result = self.dfs(new_node)
                            if result:
                                return result
                            else:
                                self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird2} to the left of {bird1} at positions {pos1 - value - 1}")
                        if pos1 + value + 1 in empty_positions:
                            new_node = deepcopy(node)
                            new_node.arrangement[bird2] = pos1 + value + 1
                            new_node.satisfied_constraints.add(current_constraint_index)
                            new_node.depth += 1
                            new_node.parent = node
                            node.children.append(new_node)
                            self.add_step(f"Current arrangement: {current_arrangement}; Trying to place {bird2} to the right of {bird1} at positions {pos1 + value + 1}")
                            result = self.dfs(new_node)
                            if result:
                                return result
                            else:
                                self.add_step(f"Current arrangement: {current_arrangement}; Cannot place {bird2} to the right of {bird1} at positions {pos1 + value + 1}")
                        else:
                            return None
        return None



def read_file(file_path: str) -> List[str]:
    """Read the puzzle file and extract the ground truth and statements."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    constraints = []

    for line in lines:
        if line[0].isdigit():
            match = re.match(r"\d+\.\s*(.+)", line.strip())
            constraints.append(match.group(1))
    # Extract ground truth and statements
    start = lines[0].index("[")
    ground_truth = lines[0][start:].strip().strip("[]").split(", ")
    ground_truth = [item.strip("'") for item in ground_truth]
    raw_constraints = constraints
    return ground_truth, raw_constraints

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager for timeout"""
    def signal_handler(signum, frame):
        raise TimeoutException("Solution took too long!")
    
    # Register signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)  # Start timer
    
    try:
        yield
    finally:
        signal.alarm(0)  # Disable timer

def solve_with_timeout(solver: PuzzleSolver, timeout_seconds: int) -> Optional[Dict[str, int]]:
    """Wrapper to call solve() with timeout"""
    try:
        with time_limit(timeout_seconds):
            return solver.solve()
    except TimeoutException:
        print(f"Solver timed out after {timeout_seconds} seconds!")
        return None

def solve_all_puzzles(num_puzzles: int):
    random.seed(0)
    timeout_seconds = 30  # Set timeout to 30 seconds
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(base_dir, "QA_sols")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    num_sols = 0
    num_timeouts = 0
    
    for i in range(num_puzzles):
        # filepath = f"generated_QAs/birds_{i}.txt"
        
        try:
            # ground_truth, raw_constraints = read_file(filepath)
            
            # if len(ground_truth) <= 7:
            print(f"Solving puzzle {i}, num_sols: {num_sols}")
            birds = deepcopy(ground_truth)
            random.shuffle(birds)
            sol_file = os.path.join(output_folder, f"sol_{i}.txt")
            
            try:
                with open(sol_file, 'w') as file:
                    sys.stdout = file
                    solver = PuzzleSolver(birds=birds, puzzle_constraints=raw_constraints)
                    solver.parse_constraints()
                    
                    start_time = time.time()
                    final_arrangement = solve_with_timeout(solver, timeout_seconds)
                    solve_time = time.time() - start_time
                    
                    if final_arrangement:
                        # print(f"Solution found in {solve_time:.2f} seconds.")
                        num_sols += 1
                    else:
                        # print(f"No solution found or timed out after {solve_time:.2f} seconds.")
                        if solve_time >= timeout_seconds:
                            num_timeouts += 1
                            # Delete the solution file for timed out puzzles
                            sys.stdout = sys.__stdout__
                            os.remove(sol_file)
                            continue
                        
            except Exception as e:
                print(f"Error solving puzzle {i}: {str(e)}")
            finally:
                sys.stdout = sys.__stdout__
                
        except Exception as e:
            print(f"Error reading puzzle {i}: {str(e)}")
            continue
    
    # print(f"Number of solvable puzzles: {num_sols}")
    # print(f"Number of timeouts: {num_timeouts}")



    
    # filepath = f"generated_QAs/birds_923.txt"
    # ground_truth, raw_constraints = read_file(filepath)
    # birds = deepcopy(ground_truth)
    # random.shuffle(birds)
    # solver = PuzzleSolver(birds=birds, puzzle_constraints= raw_constraints)
    # solver.parse_constraints()
    # final_arrangement = solver.solve()
    # if not final_arrangement:
    #     print("No solution found.")
    # else:
    #     print("Solution found.")


def solve_puzzle(ground_truth, raw_constraints):
    timeout_seconds = 30
    birds = deepcopy(ground_truth)
    random.shuffle(birds)
    solver = PuzzleSolver(birds=birds, puzzle_constraints=raw_constraints)
    solver.parse_constraints()
    start_time = time.time()
    final_arrangement = solve_with_timeout(solver, timeout_seconds)
    solve_time = time.time() - start_time
    if final_arrangement:
        return solver.steps
    else:
        return None
    


if __name__ == "__main__":
    solve_all_puzzles(1000)