EXAMPLES = """
Facts 1: Each lepidopteran is an insect. Each arthropod is a protostome. Every animal is multicellular. Protostomes are invertebrates. Each whale is bony. Each painted lady is a butterfly. Invertebrates are animals. Butterflies are lepidopterans. Each insect is six-legged. Every insect is an arthropod. Arthropods are not bony.
Query 1: True or false: Sally is not bony.
Claim 1.1: Sally is an insect.
Next 1.1: Each insect is an arthropod.
Claim 1.2: Sally is an arthropod.
Next 1.2: Arthropods are not bony.
Claim 1.3: Sally is not bony.
Next 1.3: Finish.
Claim 1.4: The query is true.

Facts 2: Prime numbers are natural numbers. Every Mersenne prime is not composite. Imaginary numbers are not real. Every real number is a number. Natural numbers are integers. Every real number is real. Every Mersenne prime is a prime number. Natural numbers are positive. Prime numbers are not composite. Integers are real numbers.
Query 2: True or false: 127 is not real.
Claim 2.1: 127 is real.
Next 2.1: Finish.
Claim 2.1: The query is false.

Facts 3: Lepidopterans are insects. Every animal is multicellular. Each insect is an arthropod. Each invertebrate is an animal. Insects are six-legged. Arthropods are small. Arthropods are invertebrates. Each butterfly is a lepidopteran. Whales are not small.
Query 3: True or false: Polly is not small.
Claim 3.1: Polly is a lepidopteran.
Next 3.2: Lepidopterans are insects.
Claim 3.2: Polly is an insect.
Next 3.2: Each insect is an arthropod.
Claim 3.3: Polly is an arthropod.
Next 3.3: Arthropods are small.
Claim 3.4: Polly is small.
Next 3.4: Finish.
Claim 3.4: The query is false.

Facts 4: Prime numbers are prime. Real numbers are numbers. Every integer is a real number. Real numbers are not imaginary. Mersenne primes are prime numbers. Complex numbers are imaginary. Each prime number is a natural number. Natural numbers are positive. Each Mersenne prime is prime. Each natural number is an integer.
Query 4: True or false: 7 is imaginary.
Claim 4.1: 7 is a natural number.
Next 4.1: Each natural number is an integer.
Claim 4.2: 7 is an integer.
Next 4.2: Every integer is a real number.
Claim 4.3: 7 is a real number.
Next 4.3: Real numbers are not imaginary.
Claim 4.4: 7 is not imaginary.
Next 4.4: Finish.
Claim 4.4: The query is false.

"""

FACTS_FORMAT = "Facts 4: {}\n"
CLAIM_FORMAT = "Claim 4.1: {}\n"
QUERY_FORMAT = "Query 4: {}\n"
NEXT_STEP_PREFIX = "Next 4.1:"
