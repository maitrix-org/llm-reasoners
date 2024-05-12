decompose_prompt = """
Q: Did the 40th president of the United States forward lolcats to his friends?
A: To answer the question "Did the 40th president of the United States forward lolcats to his friends?", we need to know: "Who was the 40th president of the United States?", "Which year did the 40th president of the United States die in?", "In what year did the first lolcat appear?", "Is the year the first lolcat appear before or the same as the year the 40th president of the United States die?".

Q: Would an uninsured person be more likely than an insured person to decline a CT scan?
A: To answer the question "Would an uninsured person be more likely than an insured person to decline a CT scan?", we need to know: "How much does it usually cost to get a CT scan without insurance?", "How much on average does it cost to get a CT scan with insurance?", "Is the cost to get a CT scan without insurance less than the cost to get a CT scan with insurance?"

Q: Were mollusks an ingredient in the color purple?
A: To answer the question "Were mollusks an ingredient in the color purple?", we need to know: "Which animal was used in making purple glands in ancient Tyre?", "Is the animal used in making purple glands in ancient Tyre a mollusk?".

Q: Is a platypus immune from cholera?
A: To answer the question "Is a platypus immune from cholera?", we need to know: "What parts of the body does Cholera damage?", "Does a platypus not have the parts of the body Cholera damage?".

Q: Are quadrupeds represented on Chinese calendar?
A: To answer the question "Are quadrupeds represented on Chinese calendar?", we need to know: "What are the symbols of the Chinese calendar?", "What is the defining characteristic of quadrapeds?", "Is any of the defining characteristic of quadrapeds included in any of the animals in the symbols of the Chinese calendar have?".
"""

solve_prompt = """
Did the 40th president of the United States forward lolcats to his friends?

Q: Who was the 40th president of the United States?
A: The 40th president of the United States was Ronald Reagan. So the answer is Ronald Reagan.

Q: Which year did the 40th president of the United States die in?
A: Ronald Reagan died in 2004. So the answer is 2004.

Q: When did the first lolcat appear?
A: The first recorded use of the term lolcat occurred in 2006. So the answer is 2006.

Q: Is the year the first lolcat appear before or the same as the year the 40th president of the United States die?
A: The first recorded use of the term lolcat occurred in 2006. The 40th president of the United States died in 2004. 2006 is not before or the same as 2004. Thus, the year the first lolcat appear is not before or the same as the year the 40th president of the United States die. So the answer is no.

Q: Did the 40th president of the United States forward lolcats to his friends?
A: The year the first lolcat appear is later than the year the 40th president of the United States die. Thus, the 40th president of the United States could not forward lolcats to his friends. So the answer is no.

Would an uninsured person be more likely than an insured person to decline a CT scan?

Q: How much does it usually cost to get a CT scan without insurance?
A: Without insurance, a CT scan can cost up to $5,000. So the answer is $5,000.

Q: How much on average does it cost to get a CT scan with insurance?
A: Most insurance companies will cover or reimburse the cost of a CT scan. So the cost will be $0. So the answer is $0.

Q: Is the cost to get a CT scan without insurance more than the cost to get a CT scan with insurance?
A: The cost to get a CT scan without insurance is $5,000. The cost to get a CT scan with insurance is $0. $5,000 is larger than $0. Thus, the cost to get a CT scan without insurance is more than the cost to get a CT scan with insurance. So the answer is yes.

Q: Would an uninsured person be more likely than an insured person to decline a CT scan?
A: The cost to get a CT scan without insurance is more than the cost to get a CT scan with insurance. Thus, an uninsured person is likely to pay more than an insured person for a CT scan. Thus, an uninsured person is more likely to to decline a CT scan than an insured person.So the answer is yes.

Is a platypus immune from cholera?
Q: What parts of the body does Cholera damage?
A: Cholera is a bacteria that damages the small intestines in humans. The intestines are part of the stomach of humans. So the answer is stomach.

Q: Does a platypus not have the parts of the body Cholera damage?
A: Cholera damage stomach. A platypus does not have a stomach. So the answer is yes. 

Q: Is a platypus immune from cholera?
A: A platypus does not have the parts of the body Cholera damage. Thus, a platypusis immune from Cholera. So the answer is yes.

Are quadrupeds represented on Chinese calendar?

Q: What are the symbols of the Chinese calendar?
A: The Chinese calendar has a number of symbols including monkeys, goats, and tigers. So the answer is a number of symbols including monkeys, goats, and tigers.

Q: What is the defining characteristic of quadrapeds?
A: Quadrupeds are animals that walk on four legs. So the answer is four legs.

Q: Is any of the defining characteristic of quadrapeds included in any of the animals in the symbols of the Chinese calendar have?
A: Chinese calendar has symbols of tigers, goats and monkeys. Tigers have four paws and balance themselves by walking on their toes. Thus, four legs, which is the defining characteristic of quadrapeds, is included by tiger. Thus, the defining characteristic of quadrapeds is included in animals in the symbols of the Chinese calendar have. So the answer is yes.

Q: Are quadrupeds represented on Chinese calendar?
A: Chinese calendar includes the defining characteristic of quadrapeds. Thus, quadrupeds are represented on Chinese calendar. So the answer is yes.
"""