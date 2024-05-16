TEMPLATE = """You are solving a logical reasoning problem. Given a query and state, your task is to decide whether a new action, i.e., a fact selected from the fact set, is a good one.

[FACTS]
Each lepidopteran is an insect. Each arthropod is a protostome. Every animal is multicellular. Protostomes are invertebrates. Each whale is bony. Each painted lady is a butterfly. Invertebrates are animals. Butterflies are lepidopterans. Every insect is six-legged. Every insect is an arthropod. Arthropods are not bony.
[QUERY]
True or false: Sally is not bony.
[STATE]
Sally is a painted lady.
[ACTION]
Each lepidopteran is an insect.
[EVALUATION] Is the last action good?
Yes, because insect is an arthropod and arthropods are not bony.

[FACTS]
Lepidopterans are insects. Every animal is multicellular. Each insect is an arthropod. Each invertebrate is an animal. Insects are six-legged. Arthropods are small. Arthropods are invertebrates. Each butterfly is a lepidopteran. Whales are not small. Polly is a lepidopteran.
[QUERY]
True or false: Polly is not small.
[STATE]
Polly is an insect.
[ACTION]
Insects are six-legged.
[EVALUATION] Is the last action good?
No, how many legs insects have is not relevant to whether Polly is small.

[FACTS]
Prime numbers are natural numbers. Every Mersenne prime is not composite. Imaginary numbers are not real. Every real number is a number. Natural numbers are integers. Every real number is real. Every Mersenne prime is a prime number. Natural numbers are positive. Prime numbers are not composite. Integers are real numbers.
[QUERY]
True or false: 127 is not real.
[STATE]
127 is a prime number.
[ACTION]
Prime numbers are prime.
[EVALUATION] Is the last action good?
No, it is not in the fact set.

[FACTS]
[[FACTS]]
[QUERY]
[[QUERY]]
[STATE]
[[STATE]]
[ACTION]
[[ACTION]]
[EVALUATION] Is the last action good?
"""