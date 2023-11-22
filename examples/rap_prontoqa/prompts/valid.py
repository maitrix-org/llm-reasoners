EXAMPLES = """
Facts: Sally is a butterfly. Butterflies are lepidopterans.
Next: Sally is a lepidopteran.
Is this reasoning step correct? Yes

Facts: Sally is a painted lady. Each painted lady is a butterfly.
Next: Sally is a butterfly.
Is this reasoning step correct? Yes

Facts: Sally is an insect. Each arthropod is a protostome.
Next: Sally is an protostome.
Is this reasoning step correct? No

Facts: Sally is an arthropod. Arthropods are not bony.
Next: Sally is not bony.
Is this reasoning step correct? Yes

Facts: Sally is a lepidopteran. Butterflies are lepidopterans.
Next: Sally is a butterfly.
Is this reasoning step correct? No

Facts: 127 is a real number. Every real number is real.
Next: 127 is real.
Is this reasoning step correct? Yes

Facts: 127 is a Mersenne prime. Every Mersenne prime is not composite.
Next: 127 is a prime number.
Is this reasoning step correct? No

Facts: 127 is a natural number. Natural numbers are integers.
Next: 127 is an integer.
Is this reasoning step correct? Yes

Facts: 127 is a prime number. Every Mersenne prime is a prime number.
Next: 127 is a Merseene prime.
Is this reasoning step correct? No

Facts: 127 is an integer. Natural numbers are positive.
Next: 127 is positive.
Is this reasoning step correct? No

Facts: Polly is a lepidopteran. Each butterfly is a lepidopteran.
Next: Polly is an Butterfly.
Is this reasoning step correct? No

Facts: Polly is an insect. Each insect is an arthropod.
Next: Polly is an arthropod.
Is this reasoning step correct? Yes

Facts: Polly is an arthropod. Arthropods are small.
Next: Polly is small.
Is this reasoning step correct? Yes

Facts: Fae is a mammal. Mammals are vertebrates.
Next: Fae is not cold-blooded.
Is this reasoning step correct? No

Facts: Fae is a cat. Every cat is a feline.
Next: Fae is a feline.
Is this reasoning step correct? Yes

Facts: Fae is a carnivore. Every feline is a carnivore.
Next: Fae is a feline.
Is this reasoning step correct? No

Facts: Fae is a feline. Every feline is a carnivore.
Next: Fae is a carnivore.
Is this reasoning step correct? Yes

Facts: 7 is a prime number. Each prime number is a natural number.
Next: 7 is a natural number.
Is this reasoning step correct? Yes

Facts: 7 is a real number. Every integer is a real number.
Next: 7 is an integer.
Is this reasoning step correct? No

Facts: 7 is a natural number. Each natural number is an integer.
Next: 7 is an integer.
Is this reasoning step correct? Yes

Facts: 7 is an integer. Integers are real numbers.
Next: 7 is real.
Is this reasoning step correct? No

Facts: Stella is a butterfly. Every butterfly is a lepidopteran.
Next: Stella is a lepidopteran.
Is this reasoning step correct? Yes

Facts: Stella in an insect. Every butterfly is a lepidopteran.
Next: Stella is lepidopteran.
Is this reasoning step correct? No

Facts: Stella is a lepidopteran. Every butterfly is a lepidopteran.
Next: Stella is a butterfly.
Is this reasoning step correct? No

"""

FACTS_FORMAT = "Facts: {} {}\n"
NEXT_STEP_FORMAT = "Next: {}\n"
VALID_PREFIX = "Is this reasoning step correct?"
