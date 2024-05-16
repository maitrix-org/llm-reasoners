EXAMPLES = """
Query: True or false: Sally is not bony.
Claim: Sally is not bony.
Are we finished? Yes

Query: True or false: Sally is not bony.
Claim: Sally is a butterfly.
Are we finished? No

Query: True or false: Sally is not bony.
Claim: Sally is six-legged.
Are we finished? No

Query: True or false: 127 is not real.
Claim: 127 is a real number.
Are we finished? No

Query: True or false: 127 is not real.
Claim: 127 is real.
Are we finished? Yes

Query: True or false: 127 is not real.
Claim: 127 is a natural number.
Are we finished? No

Query: True or false: Polly is not small.
Claim: Polly is not six-legged.
Are we finished? No

Query: True or false: Polly is not small.
Claim: Polly is small.
Are we finished? Yes

Query: True or false: Polly is not small.
Claim: Polly is multicellular.
Are we finished? No

"""

TARGET_FORMAT = "Query: {}\n"
CLAIM_FORMAT = "Claim: {}\n"
OUTPUT_PREFIX = "Are we finished?"