# 9-shot code generation
code_prompt = '''
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

# solution in Python:


def solution():
    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result





Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

# solution in Python:


def solution():
    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"""
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result





Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

# solution in Python:


def solution():
    """There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"""
    computers_initial = 9
    computers_per_day = 5
    num_days = 4  # 4 days between monday and thursday
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result





Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

# solution in Python:


def solution():
    """Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"""
    toys_initial = 5
    mom_toys = 2
    dad_toys = 2
    total_received = mom_toys + dad_toys
    total_toys = toys_initial + total_received
    result = total_toys
    return result





Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

# solution in Python:


def solution():
    """Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"""
    jason_lollipops_initial = 20
    jason_lollipops_after = 12
    denny_lollipops = jason_lollipops_initial - jason_lollipops_after
    result = denny_lollipops
    return result





Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

# solution in Python:


def solution():
    """Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"""
    leah_chocolates = 32
    sister_chocolates = 42
    total_chocolates = leah_chocolates + sister_chocolates
    chocolates_eaten = 35
    chocolates_left = total_chocolates - chocolates_eaten
    result = chocolates_left
    return result





Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

# solution in Python:


def solution():
    """If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"""
    cars_initial = 3
    cars_arrived = 2
    total_cars = cars_initial + cars_arrived
    result = total_cars
    return result





Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

# solution in Python:


def solution():
    """There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"""
    trees_initial = 15
    trees_after = 21
    trees_added = trees_after - trees_initial
    result = trees_added
    return result





Q: Tom's mother is twice as old as him. And the sum of their ages was 43 ten years ago. How old is Tom now?

# solution in Python:


def solution():
    """Tom's mother is twice as old as him. And the sum of their ages was 43 ten years ago. How old is Tom now?"""
    sum_of_ages_ten_years_ago = 43
    increase_of_sum_of_ages = 10 * 2
    sum_of_ages_now = sum_of_ages_ten_years_ago + increase_of_sum_of_ages
    age_of_tom = sum_of_ages_now / (2 + 1)
    result = age_of_tom
    return result
'''

# 5-shot (T/F)
# (https://www.mathplayground.com/wpdatabase/wpindex.html)
# (https://www.analyzemath.com/middle_school_math/grade_8/problems.html)
evaluate_prompt = '''
Q: A piece of square paper has a perimeter of 32 centimeters. Nicky's dog, Rocky, tore off 1/4 of the paper. What is the area of the remaining paper?

# solution in Python:


def solution():
    """A piece of square paper has a perimeter of 32 centimeters. Nicky's dog, Rocky, tore off 1/4 of the paper. What is the area of the remaining paper?"""
    perimeter = 32
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    fraction_torn = 1 / 4
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    area_total = (perimeter / 4) ** 2
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A), because the total area of the square can be calculated by (perimeter / 4) ** 2
    area_remaining = (1 - fraction_torn) * area_total
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    result = area_total
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (B), because the result should be area_remaining
    return result
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A), but the value of result is incorrect





Q: Ariel ran an errand for her neighbor. She went to the store and bought 24 pieces of fruit, containing oranges and bananas. She purchased three times as many oranges as bananas. On her way home, Ariel accidentally dropped twice as many oranges as bananas. She still managed to deliver 15 pieces of fruit to her neighbor. How many oranges were there?

# solution in Python:


def solution():
    """Ariel ran an errand for her neighbor. She went to the store and bought 24 pieces of fruit, containing oranges and bananas. She purchased three times as many oranges as bananas. On her way home, Ariel accidentally dropped twice as many oranges as bananas. She still managed to deliver 15 pieces of fruit to her neighbor. How many oranges were delivered?"""
    fruit_bought = 24
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    oranges_bought = 3 * fruit_bought / 5
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (B), because oranges_bought is 3 times of bananas_bought, so oranges_bought should be fruit_bought * 3 / (3 + 1)
    oranges_dropped = 2 * fruit_bought / 5
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (B), because fruit_dropped hasn't be calculated yet, and oranges_dropped should be fruit_dropped * 2 / (2 + 1)
    oranges_delivered = oranges_bought - oranges_dropped
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A), but the values of oranges_bought and oranges_dropped are incorrect
    result = oranges_delivered
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A), but the value of oranges_delivered is incorrect
    return result
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A), but the value of result is incorrect





Q: Pump A can fill a tank of water in 5 hours. Pump B can fill the same tank in 15 hours. How many minutes does it take the two pumps working together to fill the tank?

# solution in Python:


def solution():
    """Pump A can fill a tank of water in 5 hours. Pump B can fill the same tank in 15 hours. How many minutes does it take the two pumps working together to fill the tank?"""
    pump_a_hours = 5
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    pump_b_hours = 15
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    total_hours = pump_a_hours + pump_b_hours
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (B), because total_hours should be calculated based on the filling rate when the two pumps working together
    total_minutes = total_hours * 60
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A), but the value of total_hours is incorrect
    result = total_minutes
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A), but the value of total_minutes is incorrect
    return result
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A), but the value of result is incorrect





Q: Two balls A and B rotate along a circular track. Ball A makes 2 full rotations in 20 minutes. Ball B makes 5 full rotation in 10 minutes. If they start rotating now from the same point towards the same direction, how many seconds will it take for them to be at the same point they start together again?

# solution in Python:


def solution():
    """Two balls A and B rotate along a circular track. Ball A makes 2 full rotations in 20 minutes. Ball B makes 5 full rotation in 10 minutes. If they start rotating now from the same point towards the same direction, how many seconds will it take for them to be at the same point they start together again?"""
    ball_a_rotations = 2
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    ball_a_minutes = 20
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    ball_b_rotations = 5
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    ball_b_minutes = 10
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    ball_a_rotation_time = ball_a_minutes / ball_a_rotations
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    ball_b_rotation_time = ball_b_minutes / ball_b_rotations
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    time_to_meet = ball_a_rotation_time * ball_b_rotations
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (B), because the time_to_meet should be minimum positive number which is divisible by both ball_a_rotation_time and ball_b_rotation_time
    result = time_to_meet
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A), but the value of time_to_meet is incorrect
    return result
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A), but the value of result is incorrect





Q: What is the sum of the sizes of the interior angles of a polygon with 53 sides?

# solution in Python:


def solution():
    """What is the sum of the sizes of the interior angles of a polygon with 53 sides?"""
    num_sides = 53
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    interior_angles = (num_sides - 2) * 180
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A), because the formula to calculate the interior angles of an n-side polygon is (n - 2) * 180 
    result = interior_angles
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
    return result
    # Is the above line of code:
    # (A) Correct
    # (B) Incorrect
    # The above line of code is: (A)
'''

choice_prefix = ['# Is the above line of code:', '# (A) Correct', '# (B) Incorrect', '# The above line of code is:']