import random
import collections
import time
from typing import Literal, List, Set, Tuple, Callable


def format_table(header: List[str], table: List[List[str]],
                 top_format='{:^{}}', left_format=' {:<{}}', cell_format='{:<{}}',
                 col_delim=' | ', row_delim='\n', prefix_format='|', postfix_format='|'):
    table = [[''] + header + ['']] + [row for row in table]
    table_format = [[prefix_format + left_format] + len(header) * [top_format]] \
                    + len(table) * [[prefix_format + left_format] + len(header) * [cell_format]]
    col_widths = [max(len(format.format(cell, 0))
                      for format, cell in zip(col_format, col))
                  for col_format, col in zip(zip(*table_format), zip(*table))]
    return row_delim.join(
               col_delim.join(
                   format.format(cell, width)
                   for format, cell, width in zip(row_format, row, col_widths)) + f" {postfix_format}"
               for row_format, row in zip(table_format, table))


def update_range(wns: List[str], rns: List[List[Set[str]]], cmp: Callable):
    changed = False
    for rn in rns:
        classified_words = set()
        for n_col, set_of_words in enumerate(rn):
            if len(set_of_words) == 1:
                classified_words.add(next(iter(set_of_words)))
        word_to_cols = dict()
        for n_col, set_of_words in enumerate(rn):
            if len(set_of_words) != 1:
                prev_length = len(set_of_words)
                set_of_words.difference_update(classified_words)
                changed |= prev_length != len(set_of_words)
                for word in set_of_words:
                    word_to_cols.setdefault(word, set()).add(n_col)
        for word, cols in word_to_cols.items():
            if len(cols) == 1:
                x = rn[next(iter(cols))]
                if len(x) != 1:
                    x.clear()
                    x.add(word)
                    changed = True

    new_rns = [[{x for x in xs if x != wn} for xs in rn] for wn, rn in zip(wns, rns)]
    pairs = []
    for wn, rn in zip(wns, rns):
        new_pairs = []
        break_condition = True
        for cn, setn in enumerate(rn):
            if wn in setn:
                break_condition = False
                if not pairs:
                    pairs = [[]]
                for v in pairs:
                    new_pairs.append([*v, cn])
        pairs = new_pairs
        if break_condition:
            break
    for pair in pairs:
        if cmp(*pair):
            for nrn, cn, wn in zip(new_rns, pair, wns):
                nrn[cn].add(wn)
    changed |= any(rn != new_rn for rn, new_rn in zip(rns, new_rns))
    if changed:
        for rn, new_rn in zip(rns, new_rns):
            for old, new in zip(rn, new_rn):
                old.intersection_update(new)
    return changed


def update_ranges(relations: List[Tuple[List[int], List[str], Callable, ...]],
                  ranges: List[List[Set[str]]]):
    changed = False
    for ins, wns, callable_object, *_ in relations:
        changed |= update_range(wns, [ranges[i] for i in ins], callable_object)
    return changed


def generate_puzzle(table: List[List[str]], *,
                    level: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                    minimal_conditions: bool = False, max_seconds_for_minimizing: float = None,
                    tries: int = 1):
    if level not in range(1, 20 + 1):
        raise ValueError('level must be >= 1 and <= 20')

    table_wo_left = [row[1:] for row in table]
    n_attributes = len(table_wo_left)
    m_objects = len(table_wo_left[0])

    if level >= 19 and m_objects == 2:
        raise ValueError('too few objects for level >= 19')
    elif m_objects <= 1:
        raise ValueError('m_objects must be >= 2')
    elif n_attributes <= 0:
        raise ValueError('n_attributes must be >= 1')

    center = m_objects // 2
    except_flag = True
    rules_for_relations = [
        (2, lambda j1, j2: j1 == j2, ['{0}:{1} == {2}:{3}', '{2}:{3} == {0}:{1}']),
        (2, lambda j1, j2: j1 == j2 - 1, ['{0}:{1} is on the left of {2}:{3}']),
        (2, lambda j1, j2: j1 == j2 + 1, ['{0}:{1} is on the right of {2}:{3}']),
        (1, lambda j1: j1 == 0, ['{0}:{1} is on the far left']),
        (1, lambda j1, last_index=m_objects - 1: j1 == last_index, ['{0}:{1} is on the far right']),
    ] + (m_objects % 2 != 0) * [(1, lambda j1, mid=center: j1 == mid, ['{0}:{1} is in the middle'])]
    if level >= 2:
        rules_for_relations += [
            (3, lambda j1, j2, j3: j2 + 1 == j1 == j3 - 1 or j3 + 1 == j1 == j2 - 1,
             ['{0}:{1} is between {2}:{3} and {4}:{5}', '{0}:{1} is between {4}:{5} and {2}:{3}']),
        ]
    if level >= 3:
        rules_for_relations += [
            (2, lambda j1, j2: j1 == j2 - 1 or j1 == j2 + 1,
             ['{0}:{1} is on the left or right of {2}:{3}']),
            (1, lambda j1, last_index=m_objects - 1: j1 == 0 or j1 == last_index,
             ['{0}:{1} is on the far left or far right']),
        ]
    if level >= 4:
        rules_for_relations += [
            (1, lambda j1: (j1 + 1) % 2 != 0, ['{0}:{1} is in an odd position']),
            (1, lambda j1: (j1 + 1) % 2 == 0, ['{0}:{1} is in an even position']),
        ]
    if level >= 5:
        rules_for_relations += [
            (2, lambda j1, j2: j1 < j2, ['{0}:{1} is somewhere to the left of {2}:{3}']),
            (2, lambda j1, j2: j1 > j2, ['{0}:{1} is somewhere to the right of {2}:{3}']),
        ]
    if level >= 6:
        rules_for_relations += [
            (2, lambda j1, j2: j1 != j2, ['{0}:{1} != {2}:{3}', '{2}:{3} != {0}:{1}'], except_flag),
        ]
    if level >= 7:
        rules_for_relations += [
            (3, lambda j1, j2, j3: j2 < j1 < j3 or j3 < j1 < j2,
             ['{0}:{1} is somewhere between {2}:{3} and {4}:{5}',
              '{0}:{1} is somewhere between {4}:{5} and {2}:{3}']),
        ]
    if level >= 8:
        rules_for_relations += [
            (2, lambda j1, j2: j1 >= j2, ['{0}:{1} is not to the left of {2}:{3}']),
            (2, lambda j1, j2: j1 <= j2, ['{0}:{1} is not to the right of {2}:{3}']),
        ]
    if level >= 9:
        rules_for_relations += [
            (2, lambda j1, j2: j1 % 2 != j2 % 2,
             ['{0}:{1} and {2}:{3} have different parity positions',
              '{2}:{3} and {0}:{1} have different parity positions'], except_flag),
            (2, lambda j1, j2: j1 % 2 == j2 % 2,
             ['{0}:{1} and {2}:{3} have the same parity positions',
              '{2}:{3} and {0}:{1} have the same parity positions'], except_flag),
        ]
    if level >= 10:
        rules_for_relations += [
            (3, lambda j1, j2, j3: (j1 == j2 and j1 != j3) or (j1 != j2 and j1 == j3),
             ['{0}:{1} == {2}:{3} or {0}:{1} == {4}:{5}, but not both',
              '{0}:{1} == {4}:{5} or {0}:{1} == {2}:{3}, but not both'], except_flag),
            (3, lambda j1, j2, j3: (j1 == j2 and j2 != j3) or (j1 != j2 and j2 == j3),
             ['{0}:{1} == {2}:{3} or {2}:{3} == {4}:{5}, but not both',
              '{2}:{3} == {4}:{5} or {0}:{1} == {2}:{3}, but not both'], except_flag),
        ]
    if level >= 11:
        rules_for_relations += [
            (3, lambda j1, j2, j3: j1 == j2 or j1 == j3,
             ['{0}:{1} == {2}:{3} or {0}:{1} == {4}:{5} or both',
              '{0}:{1} == {4}:{5} or {0}:{1} == {2}:{3} or both'], except_flag),
            (3, lambda j1, j2, j3: j1 == j2 or j2 == j3,
             ['{0}:{1} == {2}:{3} or {2}:{3} == {4}:{5} or both',
              '{2}:{3} == {4}:{5} or {0}:{1} == {2}:{3} or both'], except_flag),
        ]
    if level >= 12:
        rules_for_relations += [
            (3, lambda j1, j2, j3: j1 != j2 or j1 != j3,
             ['{0}:{1} != {2}:{3} or {0}:{1} != {4}:{5} or both',
              '{0}:{1} != {4}:{5} or {0}:{1} != {2}:{3} or both'], except_flag),
            (3, lambda j1, j2, j3: j1 != j2 or j2 != j3,
             ['{0}:{1} != {2}:{3} or {2}:{3} != {4}:{5} or both',
              '{2}:{3} != {4}:{5} or {0}:{1} != {2}:{3} or both'], except_flag),
        ]
    if level >= 13:
        rules_for_relations.pop(0)  # pop '=='
    if level >= 14:
        rules_for_relations.pop(0)  # pop 'is on the left of'
        rules_for_relations.pop(0)  # pop 'is on the right of'
    if level >= 15:
        rules_for_relations.pop(0)  # pop 'is on the far left'
        rules_for_relations.pop(0)  # pop 'is on the far right'
        if m_objects % 2 != 0:
            rules_for_relations.pop(0)  # pop 'is in the middle'
    if level >= 16:
        rules_for_relations.pop(0)  # pop 'is between'
    if level >= 17:
        rules_for_relations.pop(0)  # pop 'is on the left or right of'
        rules_for_relations.pop(0)  # pop 'is on the far left or far right'
    if level >= 18:
        rules_for_relations.pop(0)  # pop 'is in an odd position'
        rules_for_relations.pop(0)  # pop 'is in an even position'
    if level >= 19:
        rules_for_relations.pop(0)  # pop 'is somewhere to the left of'
        rules_for_relations.pop(0)  # pop 'is somewhere to the right of'
    if level >= 20:
        rules_for_relations.pop(0)  # pop '!='
    is_minimized = False
    time_elapsed = False
    min_relations = None
    while True:
        ranges = [[set(table_wo_left[i]) for _ in range(len(table_wo_left[i]))] for i in range(len(table_wo_left))]
        relations = list()
        fail = False
        while not fail:
            needs_clarification = list()
            no_solutions = False
            solved = True
            for i, rng in enumerate(ranges):
                for j, rs in enumerate(rng):
                    if len(rs) == 0:
                        no_solutions = True
                        solved = False
                        break
                    elif len(rs) > 1:
                        solved = False
                        needs_clarification.append((i, j))
                if no_solutions:
                    break
            if solved or min_relations is not None and len(relations) >= len(min_relations):
                tries -= 1
                if min_relations is None or len(relations) < len(min_relations):
                    min_relations = relations
                if tries > 0:
                    fail = True
                    continue
            if tries <= 0:
                relations = min_relations
                if not minimal_conditions:
                    break
                number_of_relations_min = len(relations)
                number_of_relations_before = len(relations)
                start_time = time.monotonic()
                main_q = collections.deque([relations])
                while main_q:
                    current_relations = main_q.popleft()
                    for k in range(len(current_relations)):
                        new_ranges = [[set(table_wo_left[i]) for _ in range(len(table_wo_left[i]))]
                                      for i in range(len(table_wo_left))]
                        new_relations = current_relations.copy()
                        new_relations.pop(k)
                        changed = True
                        while changed:
                            changed = update_ranges(new_relations, new_ranges)

                        q = collections.deque([new_ranges])
                        possible_solutions = []
                        while q:
                            current_ranges = q.popleft()

                            no_solutions = False
                            solved = True
                            for rng in current_ranges:
                                for rs in rng:
                                    if len(rs) == 0:
                                        no_solutions = True
                                        solved = False
                                        break
                                    elif len(rs) > 1:
                                        solved = False
                                if no_solutions or not solved:
                                    break
                            if no_solutions:
                                continue
                            if solved:
                                if current_ranges not in possible_solutions:
                                    possible_solutions.append(current_ranges)
                                    if len(possible_solutions) >= 2:
                                        break
                                continue

                            for n_group, rng in enumerate(current_ranges):
                                founded = False
                                for n_x, rs in enumerate(rng):
                                    if len(rs) > 1:
                                        founded = True
                                        for r in rs:
                                            new_ranges = [[x.copy() for x in row] for row in current_ranges]
                                            new_ranges[n_group][n_x] = {r}
                                            changed = True
                                            while changed:
                                                changed = update_ranges(new_relations, new_ranges)
                                            q.appendleft(new_ranges)
                                        break
                                if founded:
                                    break
                        if len(possible_solutions) == 1:
                            number_of_relations_after = len(new_relations)
                            if number_of_relations_min > number_of_relations_after:
                                number_of_relations_min = number_of_relations_after
                                relations = new_relations
                                main_q.append(new_relations)
                        if max_seconds_for_minimizing is not None and \
                                time.monotonic() >= start_time + max_seconds_for_minimizing:
                            time_elapsed = True
                            break
                    if time_elapsed:
                        break
                is_minimized = number_of_relations_min < number_of_relations_before or not time_elapsed
                break
            if no_solutions or not needs_clarification:
                fail = True
                continue

            i, j = item = random.choice(needs_clarification)
            next2_i, next2_j = None, None
            if level >= 2 and len(needs_clarification) > 1:
                needs_clarification.remove(item)
                next2_i, next2_j = random.choice(needs_clarification)

            neighbours = []
            right_neighbours = []
            for dj in range(-1, 1 + 1):
                if not (0 <= j + dj < m_objects):
                    continue
                for new_i in range(0, n_attributes):
                    if new_i == i and dj == 0:
                        continue
                    new_item = (new_i, j + dj)
                    neighbours.append(new_item)
                    if level >= 2 and dj == 1:
                        right_neighbours.append(new_item)
            if not neighbours:
                continue
            next_i, next_j = random.choice(neighbours)
            if level >= 2 and next2_i is None and right_neighbours:
                next2_i, next2_j = random.choice(right_neighbours)

            permutations3 = [
                ((i, j), (next_i, next_j), (next2_i, next2_j)), ((i, j), (next2_i, next2_j), (next_i, next_j)),
                ((next_i, next_j), (i, j), (next2_i, next2_j)), ((next_i, next_j), (next2_i, next2_j), (i, j)),
                ((next2_i, next2_j), (i, j), (next_i, next_j)), ((next2_i, next2_j), (next_i, next_j), (i, j))
            ] if next2_i is not None else []
            permutations2 = [
                ((i, j), (next_i, next_j)), ((next_i, next_j), (next2_i, next2_j)), ((i, j), (next2_i, next2_j)),
                ((next_i, next_j), (i, j)), ((next2_i, next2_j), (next_i, next_j)), ((next2_i, next2_j), (i, j)),
            ] if next2_i is not None else [
                ((i, j), (next_i, next_j)), ((next_i, next_j), (i, j))
            ]
            possible_variants = []
            for (n_args, cmp_function, str_variants, *flags) in rules_for_relations:
                if n_args == 3:
                    for items in permutations3:
                        (ti, tj), (t_next_i, t_next_j), (t_next2_i, t_next2_j) = items
                        if flags and flags[0] and (ti == t_next_i or ti == t_next2_i or t_next_i == t_next2_i):
                            continue
                        if cmp_function(tj, t_next_j, t_next2_j):
                            possible_variants.append((n_args, items, cmp_function, random.choice(str_variants)))
                elif n_args == 2:
                    for items in permutations2:
                        (ti, tj), (t_next_i, t_next_j) = items
                        if flags and flags[0] and ti == t_next_i:
                            continue
                        if cmp_function(tj, t_next_j):
                            possible_variants.append((n_args, items, cmp_function, random.choice(str_variants)))
                elif n_args == 1 and cmp_function(j):
                    possible_variants.append((n_args, [(i, j)], cmp_function, random.choice(str_variants)))
            if not possible_variants:
                continue

            n_args, list_of_ij, cmp_function, string_format = random.choice(possible_variants)
            list_for_format = []
            ins, wns = [], []
            for i, j in list_of_ij:
                list_for_format.extend([table[i][0], table_wo_left[i][j]])
                ins.append(i)
                wns.append(table_wo_left[i][j])
            relations.append((ins, wns, cmp_function, string_format.format(*list_for_format)))

            changed = True
            while changed:
                changed = update_ranges(relations, ranges)

        if not fail:
            if minimal_conditions and not is_minimized and not time_elapsed:
                continue
            break

    premises = [t[-1] for t in relations]
    random.shuffle(premises)
    return premises

def table_to_dict(table: List[List[str]]) -> dict:
    res = {}
    number_of_rows = len(table[0])-1
    res["header"] = ["Position"]
    res["rows"] = [[str(i+1)] for i in range(number_of_rows)]
    for i, row in enumerate(table):
        res['header'].append(row[0])
        for j in range(1, len(row)):
            res['rows'][j-1].append(row[j])
    return res



def generate_table(n_attributes: int, m_objects: int) -> List[List[str]]:
    kinds_dict = {
        "Nationality": {
            "american", "argentine", "australian", "brazilian", "british",
            "canadian", "chinese", "colombian", "dutch", "egyptian",
            "french", "german", "indian", "indonesian", "italian",
            "japanese", "malaysian", "mexican", "nigerian", "pakistani",
            "polish", "russian", "spanish", "thai", "turkish",
        },
        "Food": {
            "apple", "apricot", "artichoke", "asparagus", "avocado",
            "banana", "blueberry", "broccoli", "cabbage", "carrot",
            "cauliflower", "cherry", "corn", "cranberry", "cucumber",
            "eggplant", "garlic", "grapefruit", "grapes", "kale",
            "kiwi", "lemon", "lettuce", "lime", "mango",
            "nectarine", "onion", "orange", "papaya", "peach",
            "pear", "peas", "pepper", "pineapple", "plum",
            "pomegranate", "potato", "pumpkin", "radish", "raspberry",
            "spinach", "strawberry", "tomato", "watermelon", "zucchini",
        },
        "Pet": {
            "bird", "cat", "chinchilla", "dog", "ferret",
            "fish", "frog", "goat", "goldfish", "guinea-pig",
            "hamster", "hedgehog", "horse", "lizard", "mouse",
            "pony", "rabbit", "rat", "snake", "turtle",
        },
        "Job": {
            "accountant", "analyst", "architect", "bartender", "chef",
            "coach", "dancer", "designer", "doctor", "dressmaker",
            "electrician", "engineer", "entrepreneur", "firefighter", "fisherman",
            "freelancer", "journalist", "lawyer", "librarian", "manager",
            "mechanic", "musician", "nurse", "paramedic", "photographer",
            "pilot", "police-officer", "project-manager", "scientist", "security-guard",
            "social-worker", "software-developer", "teacher", "videographer", "writer",
        },
        "Beverage": {
            "7up", "almond-milk", "coffee", "cola", "fanta",
            "hot-chocolate", "iced-tea", "juice", "lemonade", "milk",
            "mirinda", "soy-milk", "sprite", "tea", "water",
        },
        "Transport": {
            "airplane", "bike", "boat", "bus", "car",
            "helicopter", "jet-ski", "motorbike", "quad-bike", "roller",
            "scooter", "ship", "skateboard", "snowmobile",
            "subway", "taxi", "train", "tram", "trike", "van",
        },
        "Music-Genre": {
            "ambient", "blues", "classical", "country", "d&b",
            "disco", "dubstep", "electronic", "folk", "funk",
            "gospel", "hip-hop", "house", "indie", "jazz",
            "metal", "pop", "punk", "r&b", "reggae",
            "rock", "salsa", "soul", "techno", "trance",
        },
        "Movie-Genre": {
            "action", "adventure", "animation", "comedy", "crime",
            "disaster", "documentary", "drama", "epic", "family",
            "fantasy", "horror", "martial-arts", "musical", "mystery",
            "romance", "satire", "scientific", "sports", "spy",
            "superhero", "thriller", "time-travel", "western", "zombie",
        },
        "Sport": {
            "badminton", "baseball", "basketball", "biathlon", "climbing",
            "cricket", "cycling", "golf", "handball", "ice-hockey",
            "lacrosse", "parkour", "rowing", "rugby", "sailing",
            "skateboarding", "skiing", "snowboarding", "soccer", "surfing",
            "swimming", "tennis", "volleyball", "water-polo", "weightlifting",
        },
        "Hobby": {
            "baking", "board-games", "camping", "card-games", "chess",
            "collecting", "cooking", "dancing", "drawing", "filmmaking",
            "fishing", "gardening", "hiking", "magic-tricks", "photography",
            "puzzles", "reading", "rock-climbing", "singing", "skydiving",
            "sudoku", "traveling", "video-games", "woodworking", "writing",
        }
    }
    kinds = sorted(kinds_dict)

    # Check
    assert n_attributes <= len(kinds_dict),\
        f'Not enough attributes: actual {len(kinds_dict)}, expected {n_attributes}'
    assert all(m_objects <= len(v) for k, v in kinds_dict.items()), 'Not enough objects: ' +\
        f'actual {next(f"{k}={len(v)}" for k, v in kinds_dict.items() if m_objects > len(v))}, expected {m_objects}'

    chosen_kinds = sorted(random.sample(kinds, k=n_attributes))
    table = [[kind] + random.sample(sorted(kinds_dict[kind]), k=m_objects) for kind in chosen_kinds]
    return table

