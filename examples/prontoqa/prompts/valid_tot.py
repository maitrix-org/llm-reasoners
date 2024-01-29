EXAMPLES = """[FACTS]
Each lepidopteran is an insect. Each arthropod is a protostome. Every animal is multicellular. Protostomes are invertebrates. Each whale is bony. Each painted lady is a butterfly. Invertebrates are animals. Butterflies are lepidopterans. Every insect is six-legged. Every insect is an arthropod. Arthropods are not bony. Sally is a painted lady.
[QUERY]
True or false: Sally is not bony.
[ACTIONS]
"Sally is a painted lady.",
"Each painted lady is a butterfly.",
"Sally is a butterfly.",
"Butterflies are lepidopterans.",
"Sally is a lepidopteran.",
"Each lepidopteran is an insect.",
"Sally is an insect."
[EVALUATION] Is the last action good?
Yes, Sally is an insect since each lepidoteran is an insect.

[FACTS]
Prime numbers are natural numbers. Every Mersenne prime is not composite. Imaginary numbers are not real. Every real number is a number. Natural numbers are integers. Every real number is real. Every Mersenne prime is a prime number. Natural numbers are positive. Prime numbers are not composite. Integers are real numbers. 127 is a Mersenne prime.
[QUERY]
True or false: 127 is not real.
[ACTIONS]
"127 is a Mersenne prime.",
"Every Mersenne prime is a prime number.",
"127 is a prime number.",
"Prime numbers are natural numbers.",
"127 is a natural number.",
"Natural numbers are integers."
[EVALUATION] Is the last action good?
Yes, Natural numbers are integers since 127 is a natural number.

[FACTS]
Lepidopterans are insects. Every animal is multicellular. Each insect is an arthropod. Each invertebrate is an animal. Insects are six-legged. Arthropods are small. Arthropods are invertebrates. Each butterfly is a lepidopteran. Whales are not small. Polly is a lepidopteran.
[QUERY]
True or false: Polly is not small.
[ACTIONS]
"Polly is a lepidopteran.",
"Lepidopterans are insects.",
"Polly is an insect.",
"Each insect is an arthropod.",
"Polly is an arthropod.",
"Arthropods are invertebrates."
[EVALUATION] Is the last action good?
No, Arthopods are small since the query is about Polly is small or not.

[FACTS]
Every cat is a feline. Mammals are vertebrates. Bilaterians are animals. Vertebrates are chordates. Carnivores are mammals. Mammals are not cold-blooded. Each chordate is a bilaterian. Every feline is a carnivore. Snakes are cold-blooded. Animals are not unicellular. Every carnivore is not herbivorous. Fae is a cat.
[QUERY]
True or false: Fae is not cold-blooded.
[ACTIONS]
"Fae is a cat.",
"Every cat is a feline.",
"Fae is a feline.",
"Every feline is a carnivore.",
"Fae is a carnivore.",
"Every carnivore is not herbivorous."
[EVALUATION] Is the last action good?
No, Every carnivore is not herbivorous since the query is about cold-blooded.

[FACTS]
Prime numbers are prime. Real numbers are numbers. Every integer is a real number. Real numbers are not imaginary. Mersenne primes are prime numbers. Complex numbers are imaginary. Each prime number is a natural number. Natural numbers are positive. Each Mersenne prime is prime. Each natural number is an integer. 7 is a prime number.\n
[QUERY]
True or false: 7 is imaginary.
[ACTIONS]
"7 is a prime number.",
"Each prime number is a natural number.",
"7 is a natural number.",
"Each natural number is an integer.",
"7 is an integer.",
"Every integer is a real number."
[EVALUATION] Is the last action good?
Yes, Every integer is a real number since 7 is an integer.

[FACTS]
Spiders are not six-legged. Insects are six-legged. Insects are arthropods. Every animal is not unicellular. Invertebrates are animals. Lepidopterans are insects. Every arthropod is segmented. Arthropods are invertebrates. Every butterfly is a lepidopteran. Stella is a butterfly.
[QUERY]
True or false: Stella is six-legged.
[ACTIONS]
"Stella is a butterfly.",
"Every butterfly is a lepidopteran.",
"Stella is a lepidopteran.",
"Lepidopterans are insects.",
"Stella is an insect.",
"Insects are arthopods."
[EVALUATION] Is the last action good?
No, Insects are six-legged since Stella is an insect.

[FACTS]
Each natural number is not negative. Prime numbers are not composite. Mersenne primes are not composite. Real numbers are real. Real numbers are numbers. Mersenne primes are prime numbers. Integers are real numbers. Each imaginary number is not real. Every natural number is an integer. Each prime number is a natural number. 31 is a Mersenne prime.
[QUERY]
True or false: 31 is real.
[ACTIONS]
"31 is a Mersenne prime.",
"Mersenne primes are prime numbers.",
"31 is a prime number.",
"Each prime number is a natural number.",
"31 is a natural number.",
"Every natural number is an integer.",
"31 is an integer."
[EVALUATION] Is the last action good?
Yes, 31 is an integer since every natural number is an integer.

[FACTS]
Mammals are vertebrates. Carnivores are mammals. Bilaterians are animals. Vertebrates are chordates. Carnivores are not herbivorous. Tabbies are cats. Every feline is a carnivore. Chordates are bilaterians. Animals are multicellular. Mammals are warm-blooded. Snakes are not warm-blooded. Cats are felines. Sam is a tabby.
[QUERY]
True or false: Sam is warm-blooded.
[ACTIONS]
"Sam is a tabby.",
"Tabbies are cats.",
"Sam is a cat.",
"Cats are felines.",
"Sam is a cat."
[EVALUATION] Is the last action good?
No, Sam is a cat is being repeated.

"""

FACTS_FORMAT = """[FACTS]
{}
[QUERY]
True or false: {}
"""
NEXT_STEP_FORMAT = """[ACTIONS]
{}
"""
VALID_PREFIX = """[EVALUATION] Is this reasoning step correct?
"""