EXAMPLES = """
Facts: Sally is a butterfly. Butterflies are lepidopterans.
Next: Sally is a lepidopteran.

Facts: Sally is a painted lady. Each painted lady is a butterfly.
Next: Sally is a butterfly.

Facts: Sally is an insect. Every insect is an arthropod.
Next: Sally is an arthropod.

Facts: Sally is an arthropod. Arthropods are not bony.
Next: Sally is not bony.

Facts: Sally is a lepidopteran. Each lepidopteran is an insect.
Next: Sally is an insect.

Facts: 127 is a real number. Every real number is real.
Next: 127 is real.

Facts: 127 is a Mersenne prime. Every Mersenne prime is a prime number.
Next: 127 is a prime number.

Facts: 127 is a natural number. Natural numbers are integers.
Next: 127 is an integer.

Facts: 127 is a prime number. Prime numbers are natural numbers.
Next: 127 is a natural number.

Facts: 127 is an integer. Integers are real numbers.
Next: 127 is a real number.

Facts: Polly is a lepidopteran. Lepidopterans are insects.
Next: Polly is an insect.

Facts: Polly is an insect. Each insect is an arthropod.
Next: Polly is an arthropod.

Facts: Polly is an arthropod. Arthropods are small.
Next: Polly is small.

Facts: Fae is a mammal. Mammals are not cold-blooded.
Next: Fae is not cold-blooded.

Facts: Fae is a cat. Every cat is a feline.
Next: Fae is a feline.

Facts: Fae is a carnivore. Carnivores are mammals.
Next: Fae is a mammal.

Facts: Fae is a feline. Every feline is a carnivore.
Next: Fae is a carnivore.

Facts: 7 is a prime number. Each prime number is a natural number.
Next: 7 is a natural number.

Facts: 7 is a real number. Real numbers are not imaginary.
Next: 7 is not imaginary.

Facts: 7 is a natural number. Each natural number is an integer.
Next: 7 is an integer.

Facts: 7 is an integer. Integers are real numbers.
Next: 7 is a real number.

Facts: Stella is a butterfly. Every butterfly is a lepidopteran.
Next: Stella is a lepidopteran.

Facts: Stella in an insect. Insects are six-legged.
Next: Stella is six-legged.

Facts: Stella is a lepidopteran. Lepidopterans are insects.
Next: Stella is an insect.

"""

FACTS_FORMAT = "Facts: {} {}\n"
NEXT_CLAIM_PREFIX = "Next: "


